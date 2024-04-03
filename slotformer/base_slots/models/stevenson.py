import torch
from torch import nn
import torch.nn.functional as F
from nerv.training.model import BaseModel
from slotformer.base_slots.models import dVAE
from slotformer.base_slots.models.mlp import MLPDecoder


from slotformer.base_slots.models.savi import StoSAVi
from slotformer.base_slots.models.slotmixer import MLP, SlotMixerDecoder
from slotformer.base_slots.models.slotmixer_transformer import TransformerEncoder
from slotformer.base_slots.models.steve import SlotAttentionWMask
from slotformer.base_slots.models.steve_transformer import STEVETransformerDecoder
from slotformer.base_slots.models.utils import torch_cat
from slotformer.base_slots.models.vq_vae.vqvae import VQVAE


# class STEVENSON(StoSAVi):
#     """SA model with TransformerDecoder predicting patch tokens."""

#     def __init__(
#         self,
#         resolution,
#         clip_len,
#         slot_dict=dict(
#             num_slots=7,
#             slot_size=128,
#             slot_mlp_size=256,
#             num_iterations=2,
#             slots_init='shared_gaussian',
#             truncate='bi-level',
#             use_dvae_encodings = True,
#             sigma=1,
#         ),
#         vae_dict=dict(),
#         enc_dict=dict(
#             enc_channels=(3, 64, 64, 64, 64),
#             enc_ks=5,
#             enc_out_channels=128,
#             enc_norm='',
#         ),
#         dec_dict=dict(
#             dec_type='slate',
#             dec_num_layers=4,
#             dec_num_heads=4,
#             dec_d_model=128,
#             atten_type='multihead'
#         ),
#         pred_dict=dict(
#             pred_rnn=True,
#             pred_norm_first=True,
#             pred_num_layers=2,
#             pred_num_heads=4,
#             pred_ffn_dim=512,
#             pred_sg_every=None,
#         ),
#         loss_dict=dict(
#             use_img_recon_loss=False,  # dVAE decoded img recon loss
#             use_slots_correlation_loss=False,
#             use_cossine_similarity_loss=False,
#         ),
#         eps=1e-6,
#     ):
#         BaseModel.__init__(self)
        
#         self.resolution = resolution
#         self.clip_len = clip_len
#         self.eps = eps
        

#         self.slot_dict = slot_dict
#         self.vae_dict = vae_dict
#         self.enc_dict = enc_dict
#         self.dec_dict = dec_dict
#         self.pred_dict = pred_dict
#         self.loss_dict = loss_dict
        
#         # self.use_dvae_encodings = self.slot_dict['use_dvae_encodings']
        
#         self._build_slot_attention()
#         self._build_vqvae()
#         self._build_decoder()
#         self._build_predictor()
#         self._build_loss()

#         # a hack for only extracting slots
#         self.testing = False

#     def _build_slot_attention(self):
#         # Build SlotAttention module
#         # kernels x img_feats --> posterior_slots
#         self.enc_out_channels = self.enc_dict['enc_out_channels']
#         self.num_slots = self.slot_dict['num_slots']
#         self.slot_size = self.slot_dict['slot_size']
#         self.slot_mlp_size = self.slot_dict['slot_mlp_size']
#         self.num_iterations = self.slot_dict['num_iterations']
#         self.sa_truncate = self.slot_dict['truncate'] 
#         self.sa_init = self.slot_dict['slots_init'] 
#         self.sa_init_sigma = self.slot_dict['sigma']
        
#         in_features = self.vae_dict['enc_dec_dict']['z_channels']
        
#         assert self.sa_init in ['shared_gaussian', 'embedding', 'param', 'embedding_lr_sigma']
#         if self.sa_init == 'shared_gaussian':
#             self.slot_mu = nn.Parameter(torch.zeros(1, 1, self.slot_size))
#             self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, self.slot_size))
#             nn.init.xavier_uniform_(self.slot_mu)
#             nn.init.xavier_uniform_(self.slot_log_sigma)
#         elif self.sa_init == 'embedding':
#             self.slots_init = nn.Embedding(self.num_slots, self.slot_size)
#             nn.init.xavier_uniform_(self.slots_init.weight)
#         elif self.sa_init == 'embedding_lr_sigma':
#             self.slots_init = nn.Embedding(self.num_slots, self.slot_size)
#             self.sa_init_sigma = nn.Parameter(1)
#             nn.init.xavier_uniform_(self.slots_init.weight)
#         elif self.sa_init == 'param':
#             self.slots_init = nn.Parameter(
#                 nn.init.normal_(torch.empty(1, self.num_slots, self.slot_size)))
#         else:
#             raise NotImplementedError
        
#         self.slot_attention = SlotAttentionWMask(
#             in_features=in_features,
#             num_iterations=self.num_iterations,
#             num_slots=self.num_slots,
#             slot_size=self.slot_size,
#             mlp_hidden_size=self.slot_mlp_size,
#             eps=self.eps,
#             truncate=self.sa_truncate
#         )
#     def _build_vqvae(self):
#         h, w = self.resolution
#         self.visual_resolution = h // 4, w // 4
#         self.num_patches = self.visual_resolution[0] * self.visual_resolution[1]
#         # self.vocab_size
#         self.vqvae = VQVAE(
#             vq_dict=self.vae_dict['vq_dict'],
#             enc_dec_dict=self.vae_dict['enc_dec_dict'],
#             use_loss=False
#         )
#         ckp_path = self.vae_dict['ckp_path']
#         assert ckp_path, 'Please provide pretrained vqVAE weight'
#         ckp = torch.load(ckp_path, map_location='cpu')
#         if 'state_dict' in ckp:
#             ckp = ckp['state_dict']
#         ckp = {k: v for k, v in ckp.items() if 'loss' not in k}
#         self.vqvae.load_state_dict(ckp)
#         self.vqvae.freeze()

#     def _build_decoder(self):
#         # Build Decoder
#         # GPT-style causal masking Transformer decoder
#         # H, W = self.resolution
#         # self.h, self.w = H // self.down_factor, W // self.down_factor
#         # self.num_patches = self.h * self.w
#         # max_len = self.num_patches - 1
#         # self.decoder = SlotMixerDecoder(
#         #         allocator=TransformerEncoder(
#         #             **self.dec_dict['allocator'],
#         #         ),
#         #         renderer=MLP(
#         #             **self.dec_dict['renderer'],
#         #         ),
#         #         **self.dec_dict['model_conf']
#         #     )
#         self.decoder = MLPDecoder(**self.dec_dict)
            
#     def _build_loss(self):
#         """Loss calculation settings."""
#         self.use_img_recon_loss = self.loss_dict['use_img_recon_loss']
#         self.use_slots_correlation_loss = self.loss_dict['use_slots_correlation_loss']
#         self.use_cossine_similarity_loss = self.loss_dict['use_cossine_similarity_loss']
        
#     def encode(self, inp, prev_slots=None):
#         """Encode from img to slots."""
#         B, T = inp.shape[:2]
#         encoder_out = inp
#         slot_inits = None

#         # apply SlotAttn on video frames via reusing slots
#         all_slots, all_masks = [], []
#         for idx in range(T):
            
#             # init
#             if prev_slots is None:
#                 if self.sa_init == 'shared_gaussian':
#                     slot_inits = torch.randn(B, self.num_slots, self.slot_size).type_as(encoder_out) * torch.exp(self.slot_log_sigma) + self.slot_mu
#                 elif self.sa_init == 'embedding':
#                     mu = self.slots_init.weight.expand(B, -1, -1)
#                     z = torch.randn_like(mu).type_as(encoder_out)
#                     slot_inits = mu + z * self.sa_init_sigma * mu.detach()
#                 elif self.sa_init == 'param':
#                     slot_inits = self.slots_init.repeat(B, 1, 1)
    
#                 latents = slot_inits
#             else:
#                 # latents = self.predictor(prev_slots)  # [B, N, C]
#                 latents = prev_slots

#             # SA to get `post_slots`
#             slots, masks = self.slot_attention(encoder_out[:, idx], latents, slot_inits)
#             # print(masks.shape)
#             all_slots.append(slots)
#             all_masks.append(masks.unflatten(-1, self.visual_resolution))

#             # next timestep
#             prev_slots = slots

#         # (B, T, self.num_slots, self.slot_size)
#         slots = torch.stack(all_slots, dim=1)
#         # (B, T, self.num_slots, H, W)
#         masks = torch.stack(all_masks, dim=1).contiguous()

#         # resize masks to the original resolution
#         if not self.training and self.visual_resolution != self.resolution:
#             with torch.no_grad():
#                 masks = masks.flatten(0, 2).unsqueeze(1)  # [BTN, 1, H, W]
#                 masks = F.interpolate(
#                     masks,
#                     self.resolution,
#                     mode='bilinear',
#                     align_corners=False,
#                 ).squeeze(1).unflatten(0, (B, T, self.num_slots))

#         return slots, masks, encoder_out

#     def forward(self, data_dict):
#         """A wrapper for model forward.

#         If the input video is too long in testing, we manually cut it.
#         """
#         img = data_dict['img']
#         T = img.shape[1]
#         if T <= self.clip_len or self.training:
#             return self._forward(
#                 img)

#         # try to find the max len to input each time
#         clip_len = T
#         while True:
#             try:
#                 _ = self._forward(img[:, :clip_len])
#                 del _
#                 torch.cuda.empty_cache()
#                 break
#             except RuntimeError:  # CUDA out of memory
#                 clip_len = clip_len // 2 + 1
#         # update `clip_len`
#         self.clip_len = max(self.clip_len, clip_len)
#         # no need to split
#         if clip_len == T:
#             return self._forward(
#                 img)

#         # split along temporal dim
#         cat_dict = None
#         prev_slots = None
#         for clip_idx in range(0, T, clip_len):
#             out_dict = self._forward(
#                 img[:, clip_idx:clip_idx + clip_len], prev_slots=prev_slots)
#             # because this should be in test mode, we detach the outputs
#             if cat_dict is None:
#                 cat_dict = {k: [v.detach()] for k, v in out_dict.items()}
#             else:
#                 for k, v in out_dict.items():
#                     cat_dict[k].append(v.detach())
#             prev_slots = cat_dict['post_slots'][-1][:, -1].detach().clone()
#             del out_dict
#             torch.cuda.empty_cache()
#         cat_dict = {k: torch_cat(v, dim=1) for k, v in cat_dict.items()}
#         return cat_dict
    
#     def _forward(self, img, prev_slots=None):
#         """Forward function.

#         Args:
#             img: [B, T, C, H, W]
#             img_token_id: [B, T, h*w], pre-computed dVAE tokenized img ids
#             prev_slots: [B, num_slots, slot_size] or None,
#                 the `post_slots` from last timestep.
#         """
#         # reset RNN states if this is the first frame
#         if prev_slots is None:
#             self._reset_rnn()

#         B, T = img.shape[:2]
        
        
#         # tokenize the images
#         quant, _, img_token_id = self.vqvae.encode_quantize(img.flatten(0, 1))
        
        
        
#         h, w = self.visual_resolution
#         target_token_id = img_token_id.flatten(1, 2).long()  # [B*T, h*w]
#         slots_inp = quant.flatten(2).transpose(1, 2).unflatten(0, (B, T))
#         slots, slots_masks, _ = self.encode(slots_inp, prev_slots)
#         # `slots` has shape: [B, T, self.num_slots, self.slot_size]
#         # `masks` has shape: [B, T, self.num_slots, H, W]

#         out_dict = {'slots': slots, 'masks': slots_masks}
#         # if self.testing:
#         #     return out_dict
#         # Decoder token prediction loss
#         in_slots = slots.flatten(0, 1)  # [B*T, N, C]
#         pred_token_id, masks = self.decoder(in_slots)
#         masks = masks.reshape(B, T, self.num_slots, h, w)
#         if not self.training and self.visual_resolution != self.resolution:
#             with torch.no_grad():
#                 masks = masks.flatten(0, 2).unsqueeze(1)  # [BTN, 1, H, W]
#                 masks = F.interpolate(
#                     masks,
#                     self.resolution,
#                     mode='bilinear',
#                     align_corners=False,
#                 ).squeeze(1).unflatten(0, (B, T, self.num_slots))
#         # [B*T, h*w, vocab_size]
#         out_dict.update({
#             'pred_token_id': pred_token_id,
#             'target_token_id': target_token_id,
#             'decoder_masks': masks
#         })
        
#         if self.use_slots_correlation_loss or self.use_cossine_similarity_loss:
#             out_dict['slots'] = in_slots
        

#         # decode image for loss computing
#         if self.use_img_recon_loss:
#             pass
#             # out_dict['gt_img'] = img.flatten(0, 1)  # [B*T, C, H, W]
#             # logits = pred_token_id.transpose(2, 1).\
#             #     unflatten(-1, (self.h, self.w)).contiguous()  # [B*T, S, h, w]
#             # z_logits = F.log_softmax(logits, dim=1)
#             # z = gumbel_softmax(z_logits, tau=0.1, hard=False, dim=1)
#             # recon_img = self.dvae.detokenize(z)  # [B*T, C, H, W]
#             # out_dict['recon_img'] = recon_img

#         return out_dict
    
#     def _get_slots_cos_similarity(self, slots):
#         b, slots_num, _ = slots.shape
#         coss_similarity_all_mean = 0
#         for i in range(b):
#             coss_similarity_slots_mean = 0
#             for slot_idx1 in range(slots_num-1):
#                 for slot_idx2 in range(1, slots_num):
#                     coss_similarity_slots_mean = (coss_similarity_slots_mean 
#                                                   + torch.abs(torch.cosine_similarity(slots[i, slot_idx1], 
#                                                                                       slots[i, slot_idx2], dim=0))
#                                                   / slots_num)
#             coss_similarity_all_mean = coss_similarity_all_mean + coss_similarity_slots_mean        
#         return coss_similarity_all_mean
    
    
#     def _get_slots_correlation(self, slots):
#         b, slots_num, _ = slots.shape
#         corr_matrix = torch.zeros((b, slots_num, slots_num))
#         for i in range(b):
#             corr_matrix[i] = torch.abs(torch.corrcoef(slots[i])) - torch.eye(slots_num).to(slots.device)
#         return corr_matrix.mean()
    
#     def calc_train_loss(self, data_dict, out_dict):
#         """Compute loss that are general for SlotAttn models."""
#         pred_token_id = out_dict['pred_token_id'].flatten(0, -2)
#         slots = out_dict['slots']
#         target_token_id = out_dict['target_token_id'].flatten()
#         token_recon_loss = F.cross_entropy(pred_token_id, target_token_id) 
#         loss_dict = {'token_recon_loss': token_recon_loss}
#         if self.use_cossine_similarity_loss:
#             slots_similarity = self._get_slots_cos_similarity(slots)
#             loss_dict['cossine_similarity_loss'] = slots_similarity
#         if self.use_slots_correlation_loss:
#             slots_corr = self._get_slots_correlation(slots)
#             loss_dict['slots_correlation_loss'] = slots_corr
#         if self.use_img_recon_loss:
#             pass
#             # gt_img = out_dict['gt_img']
#             # recon_img = out_dict['recon_img']
#             # recon_loss = F.mse_loss(recon_img, gt_img)
#             # loss_dict['img_recon_loss'] = recon_loss
#         return loss_dict


class STEVENSON(StoSAVi):
    """SA model with TransformerDecoder predicting patch tokens."""

    def __init__(
        self,
        resolution,
        clip_len,
        slot_dict=dict(
            num_slots=7,
            slot_size=128,
            slot_mlp_size=256,
            num_iterations=2,
            slots_init='shared_gaussian',
            truncate='bi-level',
            use_dvae_encodings = True,
            sigma=1,
        ),
        vae_dict=dict(),
        enc_dict=dict(
            enc_channels=(3, 64, 64, 64, 64),
            enc_ks=5,
            enc_out_channels=128,
            enc_norm='',
        ),
        dec_dict=dict(
            dec_type='slate',
            dec_num_layers=4,
            dec_num_heads=4,
            dec_d_model=128,
            atten_type='multihead'
        ),
        pred_dict=dict(
            pred_rnn=True,
            pred_norm_first=True,
            pred_num_layers=2,
            pred_num_heads=4,
            pred_ffn_dim=512,
            pred_sg_every=None,
        ),
        loss_dict=dict(
            use_img_recon_loss=False,  # dVAE decoded img recon loss
            use_slots_correlation_loss=False,
            use_cossine_similarity_loss=False,
        ),
        eps=1e-6,
    ):
        BaseModel.__init__(self)
        
        self.resolution = resolution
        self.clip_len = clip_len
        self.eps = eps
        

        self.slot_dict = slot_dict
        self.vae_dict = vae_dict
        self.enc_dict = enc_dict
        self.dec_dict = dec_dict
        self.pred_dict = pred_dict
        self.loss_dict = loss_dict
        
        # self.use_dvae_encodings = self.slot_dict['use_dvae_encodings']
        
        self._build_slot_attention()
        self._build_vqvae()
        self._build_decoder()
        self._build_predictor()
        self._build_loss()

        # a hack for only extracting slots
        self.testing = False

    def _build_slot_attention(self):
        # Build SlotAttention module
        # kernels x img_feats --> posterior_slots
        self.enc_out_channels = self.enc_dict['enc_out_channels']
        self.num_slots = self.slot_dict['num_slots']
        self.slot_size = self.slot_dict['slot_size']
        self.slot_mlp_size = self.slot_dict['slot_mlp_size']
        self.num_iterations = self.slot_dict['num_iterations']
        self.sa_truncate = self.slot_dict['truncate'] 
        self.sa_init = self.slot_dict['slots_init'] 
        self.sa_init_sigma = self.slot_dict['sigma']
        
        in_features = self.dec_dict['dec_d_model']
        
        assert self.sa_init in ['shared_gaussian', 'embedding', 'param', 'embedding_lr_sigma']
        if self.sa_init == 'shared_gaussian':
            self.slot_mu = nn.Parameter(torch.zeros(1, 1, self.slot_size))
            self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, self.slot_size))
            nn.init.xavier_uniform_(self.slot_mu)
            nn.init.xavier_uniform_(self.slot_log_sigma)
        elif self.sa_init == 'embedding':
            self.slots_init = nn.Embedding(self.num_slots, self.slot_size)
            nn.init.xavier_uniform_(self.slots_init.weight)
        elif self.sa_init == 'embedding_lr_sigma':
            self.slots_init = nn.Embedding(self.num_slots, self.slot_size)
            self.sa_init_sigma = nn.Parameter(1)
            nn.init.xavier_uniform_(self.slots_init.weight)
        elif self.sa_init == 'param':
            self.slots_init = nn.Parameter(
                nn.init.normal_(torch.empty(1, self.num_slots, self.slot_size)))
        else:
            raise NotImplementedError
        
        self.slot_attention = SlotAttentionWMask(
            in_features=in_features,
            num_iterations=self.num_iterations,
            num_slots=self.num_slots,
            slot_size=self.slot_size,
            mlp_hidden_size=self.slot_mlp_size,
            eps=self.eps,
            truncate=self.sa_truncate
        )
    def _build_vqvae(self):
        h, w = self.resolution
        self.vocab_size = self.vae_dict['vq_dict']['n_embed']

        self.down_factor = len(self.vae_dict['enc_dec_dict']['ch_mult']) + 1
        self.visual_resolution = h // self.down_factor, w // self.down_factor
        self.num_patches = self.visual_resolution[0] * self.visual_resolution[1]
        self.vqvae = VQVAE(
            vq_dict=self.vae_dict['vq_dict'],
            enc_dec_dict=self.vae_dict['enc_dec_dict'],
            use_loss=False
        )
        ckp_path = self.vae_dict['ckp_path']
        assert ckp_path, 'Please provide pretrained vqVAE weight'
        ckp = torch.load(ckp_path, map_location='cpu')
        if 'state_dict' in ckp:
            ckp = ckp['state_dict']
        ckp = {k: v for k, v in ckp.items() if 'loss' not in k}
        self.vqvae.load_state_dict(ckp)
        self.vqvae.freeze()

    def _build_decoder(self):
        # Build Decoder
        # GPT-style causal masking Transformer decoder
        # H, W = self.resolution
        # self.h, self.w = H // self.down_factor, W // self.down_factor
        # self.num_patches = self.h * self.w
        # max_len = self.num_patches - 1
        # self.decoder = SlotMixerDecoder(
        #         allocator=TransformerEncoder(
        #             **self.dec_dict['allocator'],
        #         ),
        #         renderer=MLP(
        #             **self.dec_dict['renderer'],
        #         ),
        #         **self.dec_dict['model_conf']
        #     )
        # self.decoder = MLPDecoder(**self.dec_dict)
        H, W = self.resolution
        self.h, self.w = self.visual_resolution
        self.num_patches = self.h * self.w
        max_len = self.num_patches - 1
        self.trans_decoder = STEVETransformerDecoder(
            vocab_size=self.vocab_size,
            d_model=self.dec_dict['dec_d_model'],
            n_head=self.dec_dict['dec_num_heads'],
            max_len=max_len,
            num_slots=self.num_slots,
            num_layers=self.dec_dict['dec_num_layers'],
        )
            
    def _build_loss(self):
        """Loss calculation settings."""
        self.use_img_recon_loss = self.loss_dict['use_img_recon_loss']
        self.use_slots_correlation_loss = self.loss_dict['use_slots_correlation_loss']
        self.use_cossine_similarity_loss = self.loss_dict['use_cossine_similarity_loss']
        
    def encode(self, inp, prev_slots=None):
        """Encode from img to slots."""
        B, T = inp.shape[:2]
        encoder_out = inp
        slot_inits = None

        # apply SlotAttn on video frames via reusing slots
        all_slots, all_masks = [], []
        for idx in range(T):
            
            # init
            if prev_slots is None:
                if self.sa_init == 'shared_gaussian':
                    slot_inits = torch.randn(B, self.num_slots, self.slot_size).type_as(encoder_out) * torch.exp(self.slot_log_sigma) + self.slot_mu
                elif self.sa_init == 'embedding':
                    mu = self.slots_init.weight.expand(B, -1, -1)
                    z = torch.randn_like(mu).type_as(encoder_out)
                    slot_inits = mu + z * self.sa_init_sigma * mu.detach()
                elif self.sa_init == 'param':
                    slot_inits = self.slots_init.repeat(B, 1, 1)
    
                latents = slot_inits
            else:
                # latents = self.predictor(prev_slots)  # [B, N, C]
                latents = prev_slots

            # SA to get `post_slots`
            slots, masks = self.slot_attention(encoder_out[:, idx], latents, slot_inits)
            # print(masks.shape)
            all_slots.append(slots)
            all_masks.append(masks.unflatten(-1, self.visual_resolution))

            # next timestep
            prev_slots = slots

        # (B, T, self.num_slots, self.slot_size)
        slots = torch.stack(all_slots, dim=1)
        # (B, T, self.num_slots, H, W)
        masks = torch.stack(all_masks, dim=1).contiguous()

        # resize masks to the original resolution
        if not self.training and self.visual_resolution != self.resolution:
            with torch.no_grad():
                masks = masks.flatten(0, 2).unsqueeze(1)  # [BTN, 1, H, W]
                masks = F.interpolate(
                    masks,
                    self.resolution,
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(1).unflatten(0, (B, T, self.num_slots))

        return slots, masks, encoder_out

    def forward(self, data_dict):
        """A wrapper for model forward.

        If the input video is too long in testing, we manually cut it.
        """
        img = data_dict['img']
        T = img.shape[1]
        if T <= self.clip_len or self.training:
            return self._forward(
                img)

        # try to find the max len to input each time
        clip_len = T
        while True:
            try:
                _ = self._forward(img[:, :clip_len])
                del _
                torch.cuda.empty_cache()
                break
            except RuntimeError:  # CUDA out of memory
                clip_len = clip_len // 2 + 1
        # update `clip_len`
        self.clip_len = max(self.clip_len, clip_len)
        # no need to split
        if clip_len == T:
            return self._forward(
                img)

        # split along temporal dim
        cat_dict = None
        prev_slots = None
        for clip_idx in range(0, T, clip_len):
            out_dict = self._forward(
                img[:, clip_idx:clip_idx + clip_len], prev_slots=prev_slots)
            # because this should be in test mode, we detach the outputs
            if cat_dict is None:
                cat_dict = {k: [v.detach()] for k, v in out_dict.items()}
            else:
                for k, v in out_dict.items():
                    cat_dict[k].append(v.detach())
            prev_slots = cat_dict['post_slots'][-1][:, -1].detach().clone()
            del out_dict
            torch.cuda.empty_cache()
        cat_dict = {k: torch_cat(v, dim=1) for k, v in cat_dict.items()}
        return cat_dict
    
    def _forward(self, img, prev_slots=None):
        """Forward function.

        Args:
            img: [B, T, C, H, W]
            img_token_id: [B, T, h*w], pre-computed dVAE tokenized img ids
            prev_slots: [B, num_slots, slot_size] or None,
                the `post_slots` from last timestep.
        """
        # reset RNN states if this is the first frame
        if prev_slots is None:
            self._reset_rnn()

        B, T = img.shape[:2]
        
        
        # tokenize the images
        quant, _, img_token_id = self.vqvae.encode_quantize(img.flatten(0, 1))
        
        
        
        h, w = self.visual_resolution
        target_token_id = img_token_id.flatten(1, 2).long().detach()  # [B*T, h*w]
        slots_inp = self.trans_decoder.project_idx(target_token_id[:, :-1]).unflatten(0, (B, T)).detach()
        slots, slots_masks, _ = self.encode(slots_inp, prev_slots)
        # `slots` has shape: [B, T, self.num_slots, self.slot_size]
        # `masks` has shape: [B, T, self.num_slots, H, W]
        tokens = None

        out_dict = {'slots': slots, 'masks': slots_masks}
        # if self.testing:
        #     return out_dict
        # Decoder token prediction loss
        # in_slots = slots.flatten(0, 1)  # [B*T, N, C]
        if self.testing:
            return out_dict
        # TransformerDecoder token prediction loss
        in_slots = slots.flatten(0, 1)  # [B*T, N, C]
        in_token_id = target_token_id[:, :-1]
        pred_token_id = self.trans_decoder(in_slots, in_token_id, tokens)[:, -(h * w):]
        # [B*T, h*w, vocab_size]
        out_dict.update({
            'pred_token_id': pred_token_id,
            'target_token_id': target_token_id,
        })
        
        if self.use_slots_correlation_loss or self.use_cossine_similarity_loss:
            out_dict['slots'] = in_slots
        

        # decode image for loss computing
        if self.use_img_recon_loss:
            pass
            # out_dict['gt_img'] = img.flatten(0, 1)  # [B*T, C, H, W]
            # logits = pred_token_id.transpose(2, 1).\
            #     unflatten(-1, (self.h, self.w)).contiguous()  # [B*T, S, h, w]
            # z_logits = F.log_softmax(logits, dim=1)
            # z = gumbel_softmax(z_logits, tau=0.1, hard=False, dim=1)
            # recon_img = self.dvae.detokenize(z)  # [B*T, C, H, W]
            # out_dict['recon_img'] = recon_img

        return out_dict
    
    def _get_slots_cos_similarity(self, slots):
        b, slots_num, _ = slots.shape
        coss_similarity_all_mean = 0
        for i in range(b):
            coss_similarity_slots_mean = 0
            for slot_idx1 in range(slots_num-1):
                for slot_idx2 in range(1, slots_num):
                    coss_similarity_slots_mean = (coss_similarity_slots_mean 
                                                  + torch.abs(torch.cosine_similarity(slots[i, slot_idx1], 
                                                                                      slots[i, slot_idx2], dim=0))
                                                  / slots_num)
            coss_similarity_all_mean = coss_similarity_all_mean + coss_similarity_slots_mean        
        return coss_similarity_all_mean
    
    
    def _get_slots_correlation(self, slots):
        b, slots_num, _ = slots.shape
        corr_matrix = torch.zeros((b, slots_num, slots_num))
        for i in range(b):
            corr_matrix[i] = torch.abs(torch.corrcoef(slots[i])) - torch.eye(slots_num).to(slots.device)
        return corr_matrix.mean()
    
    def calc_train_loss(self, data_dict, out_dict):
        """Compute loss that are general for SlotAttn models."""
        pred_token_id = out_dict['pred_token_id'].flatten(0, 1)
        slots = out_dict['slots']
        target_token_id = out_dict['target_token_id'].flatten(0, 1)
        token_recon_loss = F.cross_entropy(pred_token_id, target_token_id) 
        loss_dict = {'token_recon_loss': token_recon_loss}
        if self.use_cossine_similarity_loss:
            slots_similarity = self._get_slots_cos_similarity(slots)
            loss_dict['cossine_similarity_loss'] = slots_similarity
        if self.use_slots_correlation_loss:
            slots_corr = self._get_slots_correlation(slots)
            loss_dict['slots_correlation_loss'] = slots_corr
        if self.use_img_recon_loss:
            pass
            # gt_img = out_dict['gt_img']
            # recon_img = out_dict['recon_img']
            # recon_loss = F.mse_loss(recon_img, gt_img)
            # loss_dict['img_recon_loss'] = recon_loss
        return loss_dict
