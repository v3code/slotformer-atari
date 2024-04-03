from typing import Iterable, List, Optional, Tuple, Dict, Union
import torch
from torch import nn
import timm

from slotformer.base_slots.models.utils import get_activation_fn, init_parameters




class MLP(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        outp_dim: int,
        hidden_dims: List[int],
        initial_layer_norm: bool = False,
        activation: Union[str, nn.Module] = "relu",
        final_activation: Union[bool, str] = False,
        residual: bool = False,
        weight_init: str = 'default',
    ):
        super().__init__()
        self.residual = residual
        if residual:
            assert inp_dim == outp_dim

        layers = []
        if initial_layer_norm:
            layers.append(nn.LayerNorm(inp_dim))

        cur_dim = inp_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(cur_dim, dim))
            layers.append(get_activation_fn(activation))
            cur_dim = dim

        layers.append(nn.Linear(cur_dim, outp_dim))
        if final_activation:
            if isinstance(final_activation, bool):
                final_activation = "relu"
            layers.append(get_activation_fn(final_activation))

        self.layers = nn.Sequential(*layers)
        init_parameters(self.layers, weight_init)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        outp = self.layers(inp)

        if self.residual:
            return inp + outp
        else:
            return outp

class SlotMixerDecoder(nn.Module):
    """Slot mixer decoder reconstructing jointly over all slots, but independent per position.

    Introduced in Sajjadi et al., 2022: Object Scene Representation Transformer,
    http://arxiv.org/abs/2206.06922
    """

    def __init__(
        self,
        inp_dim: int,
        outp_dim: int,
        embed_dim: int,
        n_patches: int,
        allocator: nn.Module,
        renderer: nn.Module,
        renderer_dim: Optional[int] = None,
        output_transform: Optional[nn.Module] = None,
        pos_embed_mode: Optional[str] = None,
        use_layer_norms: bool = False,
        norm_memory: bool = True,
        temperature: Optional[float] = None,
        eval_output_size: Optional[Tuple[int]] = None,
    ):
        super().__init__()
        self.allocator = allocator
        self.renderer = renderer
        self.eval_output_size = list(eval_output_size) if eval_output_size else None

        att_dim = max(embed_dim, inp_dim)
        self.scale = att_dim**-0.5 if temperature is None else temperature**-1
        self.to_q = nn.Linear(embed_dim, att_dim, bias=False)
        self.to_k = nn.Linear(inp_dim, att_dim, bias=False)

        if use_layer_norms:
            self.norm_k = nn.LayerNorm(inp_dim, eps=1e-5)
            self.norm_q = nn.LayerNorm(embed_dim, eps=1e-5)
            self.norm_memory = norm_memory
            if norm_memory:
                self.norm_memory = nn.LayerNorm(inp_dim, eps=1e-5)
            else:
                self.norm_memory = nn.Identity()
        else:
            self.norm_k = nn.Identity()
            self.norm_q = nn.Identity()
            self.norm_memory = nn.Identity()

        if output_transform is None:
            if renderer_dim is None:
                raise ValueError("Need to provide render_mlp_dim if output_transform is unspecified")
            self.output_transform = nn.Linear(renderer_dim, outp_dim)
        else:
            self.output_transform = output_transform

        if pos_embed_mode is not None and pos_embed_mode not in ("none", "add", "concat"):
            raise ValueError("If set, `pos_embed_mode` should be 'none', 'add' or 'concat'")
        self.pos_embed_mode = pos_embed_mode
        self.pos_emb = nn.Parameter(torch.randn(1, n_patches, embed_dim) * embed_dim**-0.5)
        self.init_parameters()

    def init_parameters(self):
        layers = [self.to_q, self.to_k]
        if isinstance(self.output_transform, nn.Linear):
            layers.append(self.output_transform)
        init_parameters(layers, "xavier_uniform")

    def forward(self, slots: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not self.training and self.eval_output_size is not None:
            pos_emb = timm.layers.pos_embed.resample_abs_pos_embed(
                self.pos_emb,
                new_size=self.eval_output_size,
                num_prefix_tokens=0,
            )
        else:
            pos_emb = self.pos_emb

        pos_emb = pos_emb.expand(len(slots), -1, -1)
        memory = self.norm_memory(slots)
        query_features = self.allocator(pos_emb, memory=memory)
        q = self.to_q(self.norm_q(query_features))  # B x P x D
        k = self.to_k(self.norm_k(slots))  # B x S x D

        dots = torch.einsum("bpd, bsd -> bps", q, k) * self.scale
        attn = dots.softmax(dim=-1)

        mixed_slots = torch.einsum("bps, bsd -> bpd", attn, slots)  # B x P x D

        if self.pos_embed_mode == "add":
            mixed_slots = mixed_slots + pos_emb
        elif self.pos_embed_mode == "concat":
            mixed_slots = torch.cat((mixed_slots, pos_emb), dim=-1)

        features = self.renderer(mixed_slots)
        recons = self.output_transform(features)

        return recons, attn.transpose(-2, -1)