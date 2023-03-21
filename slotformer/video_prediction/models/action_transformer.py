from typing import Union, Callable
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import GELU


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1, dtype=None, device =None):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff, dtype=dtype, device=device)
        self.w_2 = nn.Linear(d_ff, d_model, dtype=dtype, device=device)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class ActionTransformerEncoderBlock(nn.Module):
    def __init__(self, slots_dim: int, nhead: int, action_dim: int, dim_feedforward_coef: int = 4, reverse_cross_attention=False,
                 norm_first=True, dropout: float = 0.1, layer_norm_eps: float = 1e-5, batch_first: bool = False,
                 device=None, dtype=None):
        super().__init__()
        self.slots_dim = slots_dim
        self.nhead = nhead
        self.dim_feedforward_coef = dim_feedforward_coef
        self.norm_first = norm_first
        self.reverse_cross_attention = reverse_cross_attention

        self.norm_slots = nn.LayerNorm(slots_dim, eps=layer_norm_eps)
        self.norm_action = nn.LayerNorm(action_dim, eps=layer_norm_eps)

        self.attention = nn.MultiheadAttention(num_heads=nhead,
                                               embed_dim=action_dim if reverse_cross_attention else slots_dim,
                                               dropout=dropout,
                                               dtype=dtype,
                                               batch_first=batch_first,
                                               device=device,
                                               kdim=slots_dim if reverse_cross_attention else action_dim,
                                               vdim=slots_dim if reverse_cross_attention else action_dim)

        query_dim = action_dim if reverse_cross_attention else slots_dim
        self.dim_feedforward = dim_feedforward_coef * query_dim

        self.ffd = PositionwiseFeedForward(d_model=slots_dim,
                                           d_ff=self.dim_feedforward ,
                                           dtype=dtype,
                                           device=device)

    def forward(self, slots, action):
        slots = self.norm_slots(slots)
        action = self.norm_action(action)
        q, k, v = (action, slots, slots) if self.reverse_cross_attention else (slots, action, action)
        atten = self.attention(q, k, v)
        out = self.ffd(atten)
        return out


class ActionTransformerEncoder(nn.Module):
    def __init__(self, num_layers, num_heads, slots_dim, action_dim, dim_ffd_coef=4, norm_first=True,
                 dropout: float = 0.1, layer_norm_eps: float = 1e-5, batch_first: bool = False,
                 device=None, dtype=None):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.slots_dim = slots_dim
        self.action_dim = action_dim
        self.ddm_ffd_coef = dim_ffd_coef
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.norm_first = norm_first
        self.batch_first = batch_first
        self.norm = nn.LayerNorm(slots_dim, eps=layer_norm_eps)

        self.layers = self._build_layers(device, dtype)
    def _build_layers(self, device=None, dtype=None):
        # if self.num_layers < 3:
        #     raise ValueError("Should be at least")
        layers = nn.ModuleList()
        for k in range(self.num_layers):
            if k == self.num_layers - 1:
                layers.append(ActionTransformerEncoderBlock(
                    slots_dim=self.slots_dim,
                    nhead=self.num_heads,
                    action_dim=self.action_dim,
                    dim_feedforward_coef=self.ddm_ffd_coef,
                    norm_first=self.norm_first,
                    dtype=dtype,
                    device=device,
                ))
                continue

            layers.append(nn.TransformerEncoderLayer(
                d_model=self.slots_dim,
                nhead=self.num_heads,
                dim_feedforward=self.slots_dim * self.ddm_ffd_coef,
                dropout=self.dropout,
                norm_first=self.norm_first,
                batch_first=self.batch_first,
                device=device,
                dtype=dtype
            ))

        return layers

    def forward(self, slots, actions):
        for idx, layer in enumerate(self.layers):
            if idx == len(self.num_layers - 1):
                return self.layers(slots, actions)
            slots = self.layers(slots)





