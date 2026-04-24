# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE:   https://github.com/facebookresearch/mae/blob/main/models_mae.py
# SiT:   https://github.com/willisma/SiT
# --------------------------------------------------------

import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ---------------------------------------------------------------------------
#  Embedding layers
# ---------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations.
    Handles label dropout for classifier-free guidance.
    The null label index is num_classes (the extra embedding row).
    """

    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels: torch.Tensor, force_drop_ids=None) -> torch.Tensor:
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels: torch.Tensor, train: bool, force_drop_ids=None) -> torch.Tensor:
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


# ---------------------------------------------------------------------------
#  SiT / DiT blocks
# ---------------------------------------------------------------------------

class SiTBlock(nn.Module):
    """SiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """The final layer of SiT."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


# ---------------------------------------------------------------------------
#  SiT / DiT model
# ---------------------------------------------------------------------------

class SiT(nn.Module):
    """
    Scalable Interpolant Transformer (SiT) / Diffusion Transformer (DiT).

    Both architectures share this backbone; the difference lies in the
    training objective (flow-matching velocity for SiT, noise prediction
    for DiT) which is handled outside the model.
    """

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        learn_sigma: bool = True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1],
                                             int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(x.shape[0], c, h * p, h * p)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) noisy latents
            t: (B,) timesteps in [0, 1]
            y: (B,) integer class labels
        Returns:
            (B, C, H, W) predicted velocity (SiT) or noise (DiT)
        """
        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = t + y
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)
        return x


# ---------------------------------------------------------------------------
#  Positional embedding helpers
# ---------------------------------------------------------------------------

def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int,
                             cls_token: bool = False, extra_tokens: int = 0) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)
    return emb


# ---------------------------------------------------------------------------
#  Model configs
# ---------------------------------------------------------------------------

def SiT_XL_2(**kw): return SiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kw)
def SiT_XL_4(**kw): return SiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kw)
def SiT_XL_8(**kw): return SiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kw)
def SiT_L_2(**kw):  return SiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kw)
def SiT_L_4(**kw):  return SiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kw)
def SiT_L_8(**kw):  return SiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kw)
def SiT_B_2(**kw):  return SiT(depth=12, hidden_size=768,  patch_size=2, num_heads=12, **kw)
def SiT_B_4(**kw):  return SiT(depth=12, hidden_size=768,  patch_size=4, num_heads=12, **kw)
def SiT_B_8(**kw):  return SiT(depth=12, hidden_size=768,  patch_size=8, num_heads=12, **kw)
def SiT_S_2(**kw):  return SiT(depth=12, hidden_size=384,  patch_size=2, num_heads=6,  **kw)
def SiT_S_4(**kw):  return SiT(depth=12, hidden_size=384,  patch_size=4, num_heads=6,  **kw)
def SiT_S_8(**kw):  return SiT(depth=12, hidden_size=384,  patch_size=8, num_heads=6,  **kw)

SiT_models: Dict[str, type] = {
    'SiT-XL/2': SiT_XL_2, 'SiT-XL/4': SiT_XL_4, 'SiT-XL/8': SiT_XL_8,
    'SiT-L/2':  SiT_L_2,  'SiT-L/4':  SiT_L_4,  'SiT-L/8':  SiT_L_8,
    'SiT-B/2':  SiT_B_2,  'SiT-B/4':  SiT_B_4,  'SiT-B/8':  SiT_B_8,
    'SiT-S/2':  SiT_S_2,  'SiT-S/4':  SiT_S_4,  'SiT-S/8':  SiT_S_8,
    # DiT aliases (same architecture, different training objective)
    'DiT-XL/2': SiT_XL_2, 'DiT-XL/4': SiT_XL_4, 'DiT-XL/8': SiT_XL_8,
    'DiT-L/2':  SiT_L_2,  'DiT-L/4':  SiT_L_4,  'DiT-L/8':  SiT_L_8,
    'DiT-B/2':  SiT_B_2,  'DiT-B/4':  SiT_B_4,  'DiT-B/8':  SiT_B_8,
    'DiT-S/2':  SiT_S_2,  'DiT-S/4':  SiT_S_4,  'DiT-S/8':  SiT_S_8,
}
