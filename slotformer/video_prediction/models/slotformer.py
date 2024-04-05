from slotformer.video_prediction.models.mamba import Mamba, MambaArgs
import torch
import torch.nn as nn
import torch.nn.functional as F

from nerv.training import BaseModel

from slotformer.base_slots.models import StoSAVi
# from slotformer.video_prediction.models.perceiver import TransformerActionEncoderSC


def get_sin_pos_enc(seq_len, d_model):
    """Sinusoid absolute positional encoding."""
    inv_freq = 1. / (10000 ** (torch.arange(0.0, d_model, 2.0) / d_model))
    pos_seq = torch.arange(seq_len - 1, -1, -1).type_as(inv_freq)
    sinusoid_inp = torch.outer(pos_seq, inv_freq)
    pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
    return pos_emb.unsqueeze(0)  # [1, L, C]


def build_pos_enc(pos_enc, input_len, d_model):
    """Positional Encoding of shape [1, L, D]."""
    if not pos_enc:
        return None
    # ViT, BEiT etc. all use zero-init learnable pos enc
    if pos_enc == 'learnable':
        pos_embedding = nn.Parameter(torch.zeros(1, input_len, d_model))
    # in SlotFormer, we find out that sine P.E. is already good enough
    elif 'sin' in pos_enc:  # 'sin', 'sine'
        pos_embedding = nn.Parameter(
            get_sin_pos_enc(input_len, d_model), requires_grad=False)
    else:
        raise NotImplementedError(f'unsupported pos enc {pos_enc}')
    return pos_embedding


class Rollouter(nn.Module):
    """Base class for a predictor based on slot_embs."""

    def forward(self, x):
        raise NotImplementedError

    def burnin(self, x):
        pass

    def reset(self):
        pass


class SlotRollouter(Rollouter):
    """Transformer encoder only."""

    def __init__(
            self,
            num_slots,
            slot_size,
            history_len,  # burn-in steps
            t_pe='sin',  # temporal P.E.
            slots_pe='',  # slots P.E., None in SlotFormer
            # Transformer-related configs
            d_model=128,
            num_layers=4,
            num_heads=8,
            ffn_dim=512,
            norm_first=True,
            action_conditioning=False,
            discrete_actions=True,
            num_actions=12,
            actions_dim=12,
            use_teacher_forcing=False,
            mamba_state=64,
            use_mamba=False,
            cat_actions=False,
    ):
        super().__init__()

        self.num_slots = num_slots
        self.action_conditioning = action_conditioning
        self.discrete_actions = discrete_actions
        self.pred_slots = num_slots
        if self.action_conditioning:
            self.num_slots = num_slots
        self.history_len = history_len
        
        self.cat_actions = cat_actions
        
        self.use_mamba = use_mamba
        

        # if action_conditioning and discrete_actions:
        self.action_embedding = nn.Embedding(num_actions, actions_dim)
            
        
        in_dim = slot_size
        if cat_actions:
            slot_size + actions_dim

        self.in_proj = nn.Linear(in_dim, d_model)
        self.action_conditioning = action_conditioning
        self.discrete_actions = discrete_actions

        # enc_layer = nn.TransformerEncoderLayer(
        #     d_model=d_model,
        #     nhead=num_heads,
        #     dim_feedforward=ffn_dim,
        #     norm_first=norm_first,
        #     batch_first=True,
        # )

        self.rollouter = self._build_rollouter(use_mamba, d_model,
                                                           num_layers,
                                                           num_heads,
                                                           ffn_dim,
                                                           mamba_state,
                                                           norm_first)
        self.enc_t_pe = build_pos_enc(t_pe, history_len, d_model)
        self.enc_slots_pe = build_pos_enc(slots_pe, num_slots, d_model)
        # self.enc_t_act_pe = build_pos_enc('param', history_len, d_model)
        # self.out_proj = nn.Linear(d_model, slot_size)
        self.head = self._build_head(d_model, slot_size)
        
    def _build_head(self, d_model, slot_size):
        input_size = d_model*self.num_slots*self.history_len
        out_size = self.num_slots*slot_size
        
        return nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, out_size*4),
            nn.GELU(),
            nn.Linear(out_size*4, out_size)
        )
        
        
        
    def _build_rollouter(self, use_mamba = False,
                           d_model=128,
                           num_layers=4,
                           num_heads=8,
                           ffn_dim=512,
                           mamba_state=64,
                           norm_first=True):
        # if use_mamba:
        #     mamba_args = MambaArgs(
        #         d_model=d_model,
        #         n_layer=num_layers,
        #         use_head=False,
        #         d_state=mamba_state,
        #         expand=4
        #     )
        #     return Mamba(
        #         args=mamba_args
        #     )
        # elif self.action_conditioning:
        #     dec_layer = nn.TransformerDecoderLayer(
        #         d_model=d_model,
        #         nhead=num_heads,
        #         dim_feedforward=ffn_dim,
        #         norm_first=norm_first,
        #         batch_first=True,
        #     )
        #     return nn.TransformerDecoder(
        #         decoder_layer=dec_layer,
        #         num_layers=num_layers
        #     )
             
        # else:
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            norm_first=norm_first,
            batch_first=True,
        )

        return nn.TransformerEncoder(
            encoder_layer=enc_layer, num_layers=num_layers)
    
    def _project_emb(self, emb):
        B = emb.shape[0]
        emb = emb.flatten(start_dim=1)
        out = self.head(emb)
        return out.reshape(B, self.num_slots, -1)

    def forward(self, x, pred_len, actions=None, teacher_forcing=False):
        """Forward function.

        Args:
            x: [B, history_len, num_slots, slot_size]
            pred_len: int

        Returns:
            [B, pred_len, num_slots, slot_size]
        """
        max_in_steps = self.history_len
        if teacher_forcing:
            max_in_steps += pred_len
        
        # print(x.shape)
        
        assert x.shape[1] == max_in_steps, 'wrong burn-in steps'

        B = x.shape[0]
        inp = x
        if teacher_forcing:
            x = x[:, :self.history_len]
 
        if actions is not None:
            actions = actions.long()
            actions = self.action_embedding(actions)
            # in_actions = actions[:, :self.history_len].unsqueeze(2)
            if self.cat_actions:
                x = torch.cat([x, actions.unsqueeze(2).repeat(1, 1, self.num_slots, 1)], dim=-1)
            
        in_x = x.flatten(1, 2)

        # temporal_pe repeat for each slot, shouldn't be None
        # [1, T, D] --> [B, T, N, D] --> [B, T * N, D]
        enc_pe = self.enc_t_pe.unsqueeze(2). \
            repeat(B, 1, self.num_slots, 1).flatten(1, 2)
        
        # slots_pe repeat for each timestep
        if self.enc_slots_pe is not None:
            slots_pe = self.enc_slots_pe.unsqueeze(1). \
                repeat(B, self.history_len, 1, 1).flatten(1, 2)
            enc_pe = slots_pe + enc_pe

        # generate future slots autoregressively
        pred_out = []
        for i in range(pred_len):
            # project to latent space
            x = self.in_proj(in_x)
            # if actions is not None:
             # [B, T * N, slot_size]
            # encoder positional encoding
            x = x + enc_pe
            
            if self.action_conditioning:
                actions_in = actions[:, i:self.history_len+i].unsqueeze(2).repeat(1, 1, self.num_slots, 1).flatten(1, 2)
                if not self.use_mamba:
                    actions_in = actions_in + enc_pe
                    x = self.rollouter(x, actions_in)
                else:
                    x = self.rollouter(x + actions_in)
            else:
                # spatio-temporal interaction via transformer
                x = self.rollouter(x)
            # take the last N output tokens to predict slots
            pred_slots = self._project_emb(x)
            pred_out.append(pred_slots)
            # feed the predicted slots autoregressively
            if teacher_forcing:
                next_x = inp[:, self.history_len+i]
                in_x = torch.cat([in_x[:, self.num_slots:], next_x], dim=1)
            else:
                in_x = torch.cat([in_x[:, self.num_slots:], pred_out[-1]], dim=1)
            # if actions is not None: 
            #     next_idx = self.history_len + i
            #     next_action = self.action_embedding(actions[:, next_idx]).unsqueeze(1)
            #     in_x = torch.cat([in_x, next_action], dim=1)

        return torch.stack(pred_out, dim=1)

    @property
    def dtype(self):
        return self.in_proj.weight.dtype

    @property
    def device(self):
        return self.in_proj.weight.device


class SlotFormer(BaseModel):
    """Transformer-based autoregressive dynamics model over slots."""

    def __init__(
            self,
            resolution,
            clip_len,
            slot_dict=dict(
                num_slots=7,
                slot_size=128,
            ),
            dec_dict=dict(
                dec_channels=(128, 64, 64, 64, 64),
                dec_resolution=(8, 8),
                dec_ks=5,
                dec_norm='',
                dec_ckp_path='',
            ),
            rollout_dict=dict(
                num_slots=7,
                slot_size=128,
                history_len=6,
                t_pe='sin',
                slots_pe='',
                d_model=128,
                num_layers=4,
                num_heads=8,
                ffn_dim=512,
                norm_first=True,
                action_conditioning=False,
                discrete_actions=True,
                actions_dim=4,
                use_teacher_forcing=False,
                use_mamba=False,
                mamba_state=64,
            ),
            loss_dict=dict(
                rollout_len=6,
                use_img_recon_loss=False,
                use_cosine_similarity_loss=False,
                use_directional_loss=False,
                
            ),
            eps=1e-6,
    ):
        super().__init__()

        self.resolution = resolution
        self.clip_len = clip_len
        self.eps = eps
        self.use_teacher_forcing = rollout_dict.get('use_teacher_forcing', False)
        self.teacher_forcing = False

        self.slot_dict = slot_dict
        self.dec_dict = dec_dict
        self.rollout_dict = rollout_dict
        self.loss_dict = loss_dict

        self._build_slot_attention()
        self._build_decoder()
        self._build_rollouter()
        self._build_loss()

        self.testing = False  # for compatibility
        self.loss_decay_factor = 1.  # temporal loss weighting

    def _build_slot_attention(self):
        self.num_slots = self.slot_dict['num_slots']
        self.slot_size = self.slot_dict['slot_size']

    def _build_decoder(self):
        # build the same CNN decoder as in SAVi
        StoSAVi._build_decoder(self)

        # load pretrained weight
        ckp_path = self.dec_dict['dec_ckp_path']
        assert ckp_path, 'Please provide pretrained decoder weight'
        w = torch.load(ckp_path, map_location='cpu')['state_dict']
        dec_w = {k[8:]: v for k, v in w.items() if k.startswith('decoder.')}
        dec_pe_w = {
            k[22:]: v
            for k, v in w.items() if k.startswith('decoder_pos_embedding.')
        }
        self.decoder.load_state_dict(dec_w)
        self.decoder_pos_embedding.load_state_dict(dec_pe_w)

        # freeze decoder
        for p in self.decoder.parameters():
            p.requires_grad = False
        for p in self.decoder_pos_embedding.parameters():
            p.requires_grad = False
        self.decoder.eval()
        self.decoder_pos_embedding.eval()

    def _build_rollouter(self):
        """Predictor as in SAVi to transition slot from time t to t+1."""
        # Build Rollouter
        self.history_len = self.rollout_dict['history_len']  # burn-in steps
        self.rollouter = SlotRollouter(**self.rollout_dict)

    def _build_loss(self):
        """Loss calculation settings."""
        self.rollout_len = self.loss_dict['rollout_len']  # rollout steps
        self.use_img_recon_loss = self.loss_dict['use_img_recon_loss']
        self.use_cosine_similarity_loss = self.loss_dict['use_cosine_similarity_loss']
        self.use_directional_loss = self.loss_dict['use_directional_loss']
        
    def decode(self, slots):
        """Decode from slots to reconstructed images and masks."""
        # same as SAVi
        return StoSAVi.decode(self, slots)

    def rollout(self, past_slots, pred_len, decode=False, with_gt=True, actions=None,):
        """Unroll slots for `pred_len` steps, potentially decode images."""
        B = past_slots.shape[0]  # [B, T, N, C]
        if not self.teacher_forcing:
            past_slots = past_slots[:, -self.history_len:]
        pred_slots = self.rollouter(past_slots,
                                    pred_len, actions, teacher_forcing=self.teacher_forcing)
        # `decode` is usually called from outside
        # used to visualize an entire video (burn-in + rollout)
        # i.e. `with_gt` is True
        if decode:
            if with_gt:
                T = pred_len + past_slots.shape[1]
                slots = torch.cat([past_slots, pred_slots], dim=1)
            else:
                T = pred_len
                slots = pred_slots
            recon_img, recons, masks, _ = self.decode(slots.flatten(0, 1))
            out_dict = {
                'recon_combined': recon_img,  # [B*T, 3, H, W]
                'recons': recons,  # [B*T, num_slots, 3, H, W]
                'masks': masks,  # [B*T, num_slots, 1, H, W]
            }
            out_dict = {k: v.unflatten(0, (B, T)) for k, v in out_dict.items()}
            out_dict['slots'] = slots
            return out_dict
        # [B, pred_len, N, C]
        return pred_slots

    def forward(self, data_dict):
        """Forward pass."""
        slots = data_dict['slots']  # [B, T, N, C]
        actions = data_dict.get('actions')
        assert self.rollout_len + self.history_len == slots.shape[1], \
            f'wrong SlotFormer training length {slots.shape[1]}'
        if self.teacher_forcing:
            past_slots = slots[:, :self.history_len]
        else:
            past_slots = slots
        gt_slots = slots[:, self.history_len:]
        if self.use_img_recon_loss:
            out_dict = self.rollout(
                past_slots, self.rollout_len, decode=True, with_gt=False, actions=actions)
            out_dict['pred_slots'] = out_dict.pop('slots')
            out_dict['gt_slots'] = gt_slots  # both slots [B, pred_len, N, C]
        else:
            pred_slots = self.rollout(
                past_slots, self.rollout_len, decode=False, actions=actions)
            out_dict = {
                'gt_slots': gt_slots,  # both slots [B, pred_len, N, C]
                'pred_slots': pred_slots,
            }
        return out_dict

    def calc_train_loss(self, data_dict, out_dict):
        """Compute training loss."""
        loss_dict = {}
        gt_slots = out_dict['gt_slots']  # [B, rollout_T, N, C]
        pred_slots = out_dict['pred_slots']
        slots_loss = F.mse_loss(pred_slots, gt_slots, reduction='none')

        # compute per-step slot loss in eval time
        if not self.training:
            for step in range(min(6, gt_slots.shape[1])):
                loss_dict[f'slot_recon_loss_{step + 1}'] = \
                    slots_loss[:, step].mean()

        # apply temporal loss weighting as done in RPIN
        # penalize more for early steps, less for later steps
        if self.loss_decay_factor < 1.:
            w = self.loss_decay_factor ** torch.arange(gt_slots.shape[1])
            w = w.type_as(slots_loss)
            # w should sum up to rollout_T
            w = w / w.sum() * gt_slots.shape[1]
            slots_loss = slots_loss * w[None, :, None, None]

        # only compute loss on valid slots/imgs
        # e.g. in PHYRE, some videos are short, so we pad zero slots
        vid_len = data_dict.get('vid_len', None)
        trunc_loss = False
        if (vid_len is not None) and \
                (vid_len < (self.history_len + self.rollout_len)).any():
            trunc_loss = True
            valid_mask = torch.arange(gt_slots.shape[1]).to(
                gt_slots.device) + self.history_len
            valid_mask = valid_mask[None] < vid_len[:, None]  # [B, rollout_T]
            valid_mask = valid_mask.flatten(0, 1)
            slots_loss = slots_loss.flatten(0, 1)[valid_mask]
        loss_dict['slot_recon_loss'] = slots_loss.mean()

        if self.use_img_recon_loss:
            recon_combined = out_dict['recon_combined']
            gt_img = data_dict['img'][:, self.history_len:]
            imgs_loss = F.mse_loss(recon_combined, gt_img, reduction='none')
            # in case of truncated loss, we need to mask out invalid imgs
            if trunc_loss:
                imgs_loss = imgs_loss.flatten(0, 1)[valid_mask]
            loss_dict['img_recon_loss'] = imgs_loss.mean()
        return loss_dict

    @property
    def dtype(self):
        return self.rollouter.dtype

    @property
    def device(self):
        return self.rollouter.device
    
    def eval(self):
        self.teacher_forcing = False
        return super().eval()

    def train(self, mode=True):
        super().train(mode)
        # keep decoder part in eval mode
        if self.use_teacher_forcing:
            self.teacher_forcing = mode
        self.decoder.eval()
        self.decoder_pos_embedding.eval()
        return self
