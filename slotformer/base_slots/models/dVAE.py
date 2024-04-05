import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .steve_utils import Conv2dBlock, OCRConv2dBlock, conv2d, gumbel_softmax, make_one_hot

from nerv.training import BaseModel
from lpips import LPIPS

class dVAEPerceptLoss(nn.Module):
    def __init__(self, add_abs_loss=True) -> None:
        super().__init__()
        self.loss = LPIPS(net='alex')
        self.add_abs_loss = add_abs_loss
        self.freeze()
    
    def forward(self, x, recon):
        if len(x.shape) == 5:
            x = x.flatten(0, 1)
            recon = recon.flatten(0, 1)
        percept_loss = self.loss(x, recon).mean()
        out = dict(percept_loss=percept_loss)
        if self.add_abs_loss:
            out['recon_loss'] = torch.abs(x - recon).mean()
        else:
            out['recon_loss'] = F.mse_loss(x, recon)
        return out
        
    def freeze(self):
        self.eval()
        self.requires_grad_(False) 
        
        



class dVAE(BaseModel):
    """dVAE that tokenizes the input image."""

    def __init__(self, vocab_size, img_channels=3, use_ocr_version=False, use_percept_loss=False):
        super().__init__()

        self.vocab_size = vocab_size
        self.img_channels = img_channels
        self.tau = 1.

        self.use_ocr_version = use_ocr_version
        self.convblock = OCRConv2dBlock if self.use_ocr_version else Conv2dBlock
        
        self._build_encoder()
        self._build_decoder()
        self.use_percept_loss = use_percept_loss
        if use_percept_loss:
            self.loss = dVAEPerceptLoss()
        

        # a hack for only extracting tokens
        self.testing = False

    def _build_encoder(self):
        self.encoder = nn.Sequential(
            self.convblock(self.img_channels, 64, 4, 4),
            self.convblock(64, 64, 1, 1),
            self.convblock(64, 64, 1, 1),
            self.convblock(64, 64, 1, 1),
            self.convblock(64, 64, 1, 1),
            self.convblock(64, 64, 1, 1),
            self.convblock(64, 64, 1, 1),
            conv2d(64, self.vocab_size, 1),
        )

    def _build_decoder(self):
        self.decoder = nn.Sequential(
            self.convblock(self.vocab_size, 64, 1),
            self.convblock(64, 64, 3, 1, 1),
            self.convblock(64, 64, 1, 1),
            self.convblock(64, 64, 1, 1),
            self.convblock(64, 256, 1),
            nn.PixelShuffle(2),
            self.convblock(64, 64, 3, 1, 1),
            self.convblock(64, 64, 1, 1),
            self.convblock(64, 64, 1, 1),
            self.convblock(64, 256, 1),
            nn.PixelShuffle(2),
            conv2d(64, self.img_channels, 1),
        )

    def tokenize(self, imgs, one_hot=True, return_logits=False):
        """Tokenize images."""
        B = imgs.shape[0]
        # [B, T, C, H, W]
        if len(imgs.shape) == 5:
            unflatten = True
            imgs = imgs.flatten(0, 1)
        # [B, C, H, W]
        else:
            unflatten = False

        # encode, [B, vocab_size, h, w]
        logits = self.encoder(imgs)

        # one-hot encoding, [B, vocab_size, h, w]
        if one_hot:
            z_hard = make_one_hot(logits, dim=1)
        # directly take the argmax index, [B, h, w]
        else:
            z_hard = logits.argmax(dim=1)

        if unflatten:
            z_hard = z_hard.unflatten(0, (B, -1))
            
        if return_logits:
            return z_hard, logits
        return z_hard

    def detokenize(self, z):
        """Decode z to reconstruct image.

        z should be `vocab_size` probabilities.
        """
        assert z.shape[-3] == self.vocab_size
        B = z.shape[0]
        # [B, T, vocab_size, h, w]
        if len(z.shape) == 5:
            unflatten = True
            z = z.flatten(0, 1)
        # [B, vocab_size, h, w]
        else:
            unflatten = False

        # decode, [B, C, H, W]
        recon = self.decoder(z)

        if unflatten:
            recon = recon.unflatten(0, (B, -1))

        return recon

    def forward(self, data_dict):
        """Forward function.

        Args:
            data_dict:
                - img: [B, (T, )C, H, W], input image
                - gumbel_tau: float, temperature, defaults as `self.tau`
                - hard: use STE for sampling z, default as False
        """
        if self.testing:
            return self.tokenize(data_dict['img'], one_hot=False)

        x = data_dict['img']
        tau = data_dict.get('gumbel_tau', self.tau)
        hard = data_dict.get('hard', False)

        B = x.shape[0]
        if len(x.shape) == 5:
            unflatten = True
            x = x.flatten(0, 1)
        else:
            unflatten = False

        # encode, [B, vocab_size, h, w]
        logits = self.encoder(x)
        z_logits = F.log_softmax(logits, dim=1)
        
        # sample z, [B, vocab_size, h, w]
        z = gumbel_softmax(z_logits, tau, hard=hard, dim=1)

        # decode, [B, C, H, W]
        recon = self.decoder(z)

        if unflatten:
            recon = recon.unflatten(0, (B, -1))
            z_logits = z_logits.unflatten(0, (B, -1))

        out_dict = {'recon': recon, 'z_logits': z_logits}
        return out_dict

    def calc_train_loss(self, data_dict, out_dict):
        """Compute training loss."""
        img = data_dict['img']
        recon = out_dict['recon']
        if self.use_percept_loss: 
            return self.loss(img, recon)
        recon_loss = F.mse_loss(recon, img)
        return {'recon_loss': recon_loss}
    
    @torch.no_grad()
    def calc_eval_loss(self, data_dict, out_dict):
        """Loss computation in eval."""
        loss_dict = self.calc_train_loss(data_dict, out_dict)
        # if self.use_percept_loss:
        #     img = data_dict['img']
        #     recon = out_dict['recon']
        #     loss_dict['recon_mse'] = F.mse_loss(recon, img)
        return loss_dict
    @property
    def dtype(self):
        return self.encoder[-1].weight.dtype

    @property
    def device(self):
        return self.encoder[-1].weight.device
