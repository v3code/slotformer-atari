from typing import Iterable, Union
import torch
import torch.nn as nn


def get_lr(optimizer):
    """Get the learning rate of current optimizer."""
    return optimizer.param_groups[0]['lr']


def torch_stack(tensor_list, dim):
    if len(tensor_list[0].shape) < dim:
        return torch.stack(tensor_list)
    return torch.stack(tensor_list, dim=dim)


def torch_cat(tensor_list, dim):
    if len(tensor_list[0].shape) <= dim:
        return torch.cat(tensor_list)
    return torch.cat(tensor_list, dim=dim)


def clip_tensor_norm(tensor, norm, dim=-1, eps=1e-6):
    """Clip the norm of tensor along `dim`."""
    assert norm > 0.
    tensor_norm = torch.norm(tensor, p=2, dim=dim, keepdim=True)
    scale_factor = norm / (tensor_norm + eps)
    scale_factor = torch.clip(scale_factor, max=1.)
    clip_tensor = tensor * scale_factor
    return clip_tensor


def assert_shape(actual, expected, message=""):
    assert list(actual) == list(expected), \
        f"Expected shape: {expected} but passed shape: {actual}. {message}"


def build_grid(resolution):
    """return grid with shape [1, H, W, 4]."""
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges, indexing='ij')
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    grid = grid.unsqueeze(0)
    return torch.cat([grid, 1.0 - grid], dim=-1)


def to_rgb_from_tensor(x):
    """Reverse the Normalize operation in torchvision."""
    return (x * 0.5 + 0.5).clamp(0, 1)


class SoftPositionEmbed(nn.Module):
    """Soft PE mapping normalized coords to feature maps."""

    def __init__(self, hidden_size, resolution):
        super().__init__()
        self.dense = nn.Linear(in_features=4, out_features=hidden_size)
        self.register_buffer('grid', build_grid(resolution))  # [1, H, W, 4]

    def forward(self, inputs):
        """inputs: [B, C, H, W]."""
        emb_proj = self.dense(self.grid).permute(0, 3, 1, 2).contiguous()
        return inputs + emb_proj

def get_activation_fn(name_or_instance: Union[str, nn.Module]) -> nn.Module:
    if isinstance(name_or_instance, nn.Module):
        return name_or_instance
    elif isinstance(name_or_instance, str):
        if name_or_instance.lower() == "relu":
            return nn.ReLU(inplace=True)
        elif name_or_instance.lower() == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unknown activation function {name_or_instance}")
    else:
        raise ValueError(
            f"Unsupported type for activation function: {type(name_or_instance)}. "
            "Can be `str` or `torch.nn.Module`."
        )


def init_parameters(layers: Union[nn.Module, Iterable[nn.Module]], weight_init: str = "default"):
    assert weight_init in ("default", "he_uniform", "he_normal", "xavier_uniform", "xavier_normal")
    if isinstance(layers, nn.Module):
        layers = [layers]

    for idx, layer in enumerate(layers):
        if hasattr(layer, "bias") and layer.bias is not None:
            nn.init.zeros_(layer.bias)

        if hasattr(layer, "weight") and layer.weight is not None:
            gain = 1.0
            if isinstance(layers, nn.Sequential):
                if idx < len(layers) - 1:
                    next = layers[idx + 1]
                    if isinstance(next, nn.ReLU):
                        gain = 2**0.5

            if weight_init == "he_uniform":
                torch.nn.init.kaiming_uniform_(layer.weight, gain)
            elif weight_init == "he_normal":
                torch.nn.init.kaiming_normal_(layer.weight, gain)
            elif weight_init == "xavier_uniform":
                torch.nn.init.xavier_uniform_(layer.weight, gain)
            elif weight_init == "xavier_normal":
                torch.nn.init.xavier_normal_(layer.weight, gain)
                
class LayerScale(nn.Module):
    """Module scaling input by learned scalar.

    Adapted from timm library.
    """

    def __init__(self, dim: int, init_values: float = 1e-5, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

