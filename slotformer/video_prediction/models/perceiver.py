from typing import Optional
import torch
import torch.nn as nn
from einops import rearrange


class Sequential(nn.Sequential):
    def forward(self, *x, **kwargs):
        for i, module in enumerate(self):
            if type(x) == tuple:
                if i == 0:
                    x = module(*x, **kwargs)
                else:
                    x = module(*x)
            else:
                x = module(x)
        return x


class Residual(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs) + args[0]


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            num_heads: int,
            num_q_input_channels: int,
            num_kv_input_channels: int,
            num_qk_channels: Optional[int] = None,
            num_v_channels: Optional[int] = None,
            num_output_channels: Optional[int] = None,
            max_heads_parallel: Optional[int] = None,
            causal_attention: bool = False,
            dropout: float = 0.0,
            qkv_bias: bool = True,
            out_bias: bool = True,
    ):
        """Multi-head attention as specified in https://arxiv.org/abs/2107.14795 Appendix E plus support for rotary
        position embeddings (https://arxiv.org/abs/2104.09864) and causal attention. Causal attention requires
        queries and keys to be right-aligned, if they have different length.
        :param num_heads: Number of attention heads.
        :param num_q_input_channels: Number of query input channels.
        :param num_kv_input_channels: Number of key/value input channels.
        :param num_qk_channels: Number of query and key channels. Default is number `num_q_input_channels`
        :param num_v_channels: Number of value channels. Default is `num_qk_channels`.
        :param num_output_channels: Number of output channels. Default is `num_q_input_channels`
        :param max_heads_parallel: Maximum number of heads to be processed in parallel. Default is `num_heads`.
        :param causal_attention: Whether to apply a causal attention mask. Default is `False`.
        :param dropout: Dropout probability for attention matrix values. Default is `0.0`
        :param qkv_bias: Whether to use a bias term for query, key and value projections. Default is `True`.
        :param qkv_bias: Whether to use a bias term for output projection. Default is `True`.
        """
        super().__init__()

        if num_qk_channels is None:
            num_qk_channels = num_q_input_channels

        if num_v_channels is None:
            num_v_channels = num_qk_channels

        if num_output_channels is None:
            num_output_channels = num_q_input_channels

        if num_qk_channels % num_heads != 0:
            raise ValueError("num_qk_channels must be divisible by num_heads")

        if num_v_channels % num_heads != 0:
            raise ValueError("num_v_channels must be divisible by num_heads")

        num_qk_channels_per_head = num_qk_channels // num_heads

        self.dp_scale = num_qk_channels_per_head ** -0.5
        self.num_heads = num_heads
        self.causal_attention = causal_attention

        if max_heads_parallel is None:
            self.max_heads_parallel = num_heads
        else:
            self.max_heads_parallel = max_heads_parallel

        self.q_proj = nn.Linear(num_q_input_channels, num_qk_channels, bias=qkv_bias)
        self.k_proj = nn.Linear(num_kv_input_channels, num_qk_channels, bias=qkv_bias)
        self.v_proj = nn.Linear(num_kv_input_channels, num_v_channels, bias=qkv_bias)
        self.o_proj = nn.Linear(num_v_channels, num_output_channels, bias=out_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x_q,
            x_kv,
            pad_mask=None,
            rot_pos_emb_q: Optional[RotaryPositionEmbedding] = None,
            rot_pos_emb_k: Optional[RotaryPositionEmbedding] = None,
    ):
        """
        :param x_q: Query input of shape (B, N, D) where B is the batch size, N the query sequence length
            and D the number of query input channels (= `num_q_input_channels`)
        :param x_kv: Key/value input of shape (B, L, C) where B is the batch size, L the key/value sequence
            length and C are the number of key/value input channels (= `num_kv_input_channels`)
        :param pad_mask: Boolean key padding mask. `True` values indicate padding tokens.
        :param rot_pos_emb_q: Applies a rotary position embedding to query i.e. if defined, rotates the query.
        :param rot_pos_emb_k: Applies a rotary position embedding to key i.e. if defined, rotates the key.
        :return: attention result of shape (B, N, F) where B is the batch size, N the query sequence length
            and F the number of output channels (= `num_output_channels`)
        """

        q = self.q_proj(x_q)
        k = self.k_proj(x_kv)
        v = self.v_proj(x_kv)

        q, k, v = (rearrange(x, "b n (h c) -> b h n c", h=self.num_heads) for x in [q, k, v])
        q = q * self.dp_scale

        if rot_pos_emb_q is not None:
            q = rot_pos_emb_q.rotate(q)

        if rot_pos_emb_k is not None:
            k = rot_pos_emb_k.rotate(k)

        if pad_mask is not None:
            pad_mask = rearrange(pad_mask, "b j -> b 1 1 j")

        if self.causal_attention:
            i = q.shape[2]
            j = k.shape[2]

            # If q and k have different length, causal masking only works if they are right-aligned.
            causal_mask = torch.ones((i, j), device=x_q.device, dtype=torch.bool).triu(j - i + 1)

        o_chunks = []

        # Only process a given maximum number of heads in
        # parallel, using several iterations, if necessary.
        for q_chunk, k_chunk, v_chunk in zip(
                q.split(self.max_heads_parallel, dim=1),
                k.split(self.max_heads_parallel, dim=1),
                v.split(self.max_heads_parallel, dim=1),
        ):
            attn = torch.einsum("b h i c, b h j c -> b h i j", q_chunk, k_chunk)
            attn_max_neg = -torch.finfo(attn.dtype).max

            if pad_mask is not None:
                attn.masked_fill_(pad_mask, attn_max_neg)

            if self.causal_attention:
                attn.masked_fill_(causal_mask, attn_max_neg)

            attn = attn.softmax(dim=-1)
            attn = self.dropout(attn)

            o_chunk = torch.einsum("b h i j, b h j c -> b h i c", attn, v_chunk)
            o_chunks.append(o_chunk)

        o = torch.cat(o_chunks, dim=1)
        o = rearrange(o, "b h n c -> b n (h c)", h=self.num_heads)

        return self.o_proj(o)


class CrossAttention(nn.Module):
    def __init__(
            self,
            num_heads: int,
            num_q_input_channels: int,
            num_kv_input_channels: int,
            num_qk_channels: Optional[int] = None,
            num_v_channels: Optional[int] = None,
            max_heads_parallel: Optional[int] = None,
            causal_attention: bool = False,
            dropout: float = 0.0,
            qkv_bias: bool = True,
            out_bias: bool = True,
    ):
        """Pre-layer-norm cross-attention (see `MultiHeadAttention` for attention details)."""
        super().__init__()
        self.q_norm = nn.LayerNorm(num_q_input_channels)
        self.kv_norm = nn.LayerNorm(num_kv_input_channels)
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            num_q_input_channels=num_q_input_channels,
            num_kv_input_channels=num_kv_input_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            max_heads_parallel=max_heads_parallel,
            causal_attention=causal_attention,
            dropout=dropout,
            qkv_bias=qkv_bias,
            out_bias=out_bias,
        )

    def forward(self, x_q, x_kv=None, x_kv_prefix=None, pad_mask=None, rot_pos_emb_q=None, rot_pos_emb_k=None):
        """Pre-layer-norm cross-attention of query input `x_q` to key/value input (`x_kv` or `x_kv_prefix`).
        If `x_kv_prefix` is defined, the entire key/value input is a concatenation of `x_kv_prefix` and `x_q` along
        the sequence dimension. In this case, the query attends to itself at the end of the key/value sequence (use
        case: Perceiver AR). If `x_kv_prefix` is not defined, `x_kv` is the entire key/value input.
        """
        x_q = self.q_norm(x_q)

        if x_kv is None:
            x_kv_prefix = self.kv_norm(x_kv_prefix)
            x_kv = torch.cat([x_kv_prefix, x_q], dim=1)
        else:
            x_kv = self.kv_norm(x_kv)

        return self.attention(x_q, x_kv, pad_mask=pad_mask, rot_pos_emb_q=rot_pos_emb_q, rot_pos_emb_k=rot_pos_emb_k)


class SelfAttention(nn.Module):
    def __init__(
            self,
            num_heads: int,
            num_channels: int,
            num_qk_channels: Optional[int] = None,
            num_v_channels: Optional[int] = None,
            max_heads_parallel: Optional[int] = None,
            causal_attention: bool = False,
            dropout: float = 0.0,
            qkv_bias: bool = True,
            out_bias: bool = True,
    ):
        """Pre-layer norm self-attention (see `MultiHeadAttention` and for attention details)."""
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            num_q_input_channels=num_channels,
            num_kv_input_channels=num_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            max_heads_parallel=max_heads_parallel,
            causal_attention=causal_attention,
            dropout=dropout,
            qkv_bias=qkv_bias,
            out_bias=out_bias,
        )

    def forward(self, x, pad_mask=None, rot_pos_emb=None):
        """Pre-layer-norm self-attention of input `x`."""
        x = self.norm(x)
        return self.attention(x, x, pad_mask=pad_mask, rot_pos_emb_q=rot_pos_emb, rot_pos_emb_k=rot_pos_emb)


class CrossAttentionLayer(Sequential):
    def __init__(
            self,
            num_heads: int,
            num_q_input_channels: int,
            num_kv_input_channels: int,
            num_qk_channels: Optional[int] = None,
            num_v_channels: Optional[int] = None,
            max_heads_parallel: Optional[int] = None,
            causal_attention: bool = False,
            ffn_dim: int = 528,
            dropout: float = 0.0,
            attention_residual: bool = True,
            qkv_bias: bool = True,
            out_bias: bool = True,
            mlp_bias: bool = True,
    ):
        cross_attn = CrossAttention(
            num_heads=num_heads,
            num_q_input_channels=num_q_input_channels,
            num_kv_input_channels=num_kv_input_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            max_heads_parallel=max_heads_parallel,
            causal_attention=causal_attention,
            dropout=dropout,
            qkv_bias=qkv_bias,
            out_bias=out_bias,
        )
        super().__init__(
            Residual(cross_attn) if attention_residual else cross_attn,
            Residual(MLP(num_q_input_channels, ffn_dim, bias=mlp_bias)),
        )


class SelfAttentionLayer(Sequential):
    def __init__(
            self,
            num_heads: int,
            num_channels: int,
            num_qk_channels: Optional[int] = None,
            num_v_channels: Optional[int] = None,
            max_heads_parallel: Optional[int] = None,
            causal_attention: bool = False,
            ffn_dim: int = 528,
            dropout: float = 0.0,
            qkv_bias: bool = True,
            out_bias: bool = True,
            mlp_bias: bool = True,
    ):
        self_attn = SelfAttention(
            num_heads=num_heads,
            num_channels=num_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            max_heads_parallel=max_heads_parallel,
            causal_attention=causal_attention,
            dropout=dropout,
            qkv_bias=qkv_bias,
            out_bias=out_bias,
        )
        super().__init__(
            Residual(self_attn),
            Residual(MLP(num_channels, ffn_dim, bias=mlp_bias)),
        )


class MLP(Sequential):
    def __init__(self, num_channels: int, ffn_dim: int, bias: bool = True):
        super().__init__(
            nn.LayerNorm(num_channels),
            nn.Linear(num_channels, ffn_dim, bias=bias),
            nn.GELU(),
            nn.Linear(ffn_dim, num_channels, bias=bias),
        )


class TransformerActionEncoderSC(nn.Module):
    def __init__(self,
                 d_model: int,
                 actions_dim: int,
                 ffn_dim: int,
                 num_heads: int,
                 dropout: float = 0,
                 num_layers: int = 1):
        self.num_layers = num_layers

        layers = []
        for _ in range(self.num_layers):
            layers.append(SelfAttentionLayer(num_heads=num_heads,
                                             ffn_dim=ffn_dim,
                                             num_channels=d_model,
                                             dropout=dropout,
                                             ))
            layers.append(CrossAttentionLayer(num_heads=num_heads,
                                              ffn_dim=ffn_dim,
                                              num_q_input_channels=d_model,
                                              num_kv_input_channels=actions_dim,
                                              dropout=dropout, ))

        self.layers = nn.ModuleList(layers)

    def forward(self, x, actions):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0:
                x = layer(x)
            else:
                x = layer(x, actions)
