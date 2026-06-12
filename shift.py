# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DiT：https://github.com/facebookresearch/DiT
# ControlNet: https://github.com/lllyasviel/ControlNet
# --------------------------------------------------------
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from timm.layers.drop import DropPath

from colorama import Fore

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps                                  #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        sinusoidal timestep embeddings.
        """
        half        = dim // 2
        freqs       = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args        = t[:, None].float() * freqs[None]
        embedding   = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        """
        Args:
            t:(B,)
        Returns:
            t_emb:(B,hidden_size)
        """
        t_freq  = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb   = self.mlp(t_freq)
        return t_emb


#################################################################################
#             Learnable Upsampling Encoder (In-Network Upsampling)              #
#################################################################################

class LearnableUpsampleEncoder(nn.Module):
    """
    learnable upsampling encoder.
    """
    def __init__(self, in_channels, hidden_size, patch_size):
        super().__init__()
        self.patch_size = patch_size
        
        # 4x Upsampling Network
        self.upsample_net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            
            nn.Conv2d(64, 64 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.SiLU(),
            
            nn.Conv2d(64, 64 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.SiLU(),

            # nn.Conv2d(64, 64 * 4, kernel_size=3, padding=1),
            # nn.PixelShuffle(2),
            # nn.SiLU(),
            
            nn.Conv2d(64, hidden_size, kernel_size=3, padding=1)
        )
        
        # Zero Conv for safe injection
        self.zero_conv = nn.Conv2d(hidden_size, hidden_size, kernel_size=1)
        nn.init.constant_(self.zero_conv.weight, 0)
        nn.init.constant_(self.zero_conv.bias, 0)

    def forward(self, x, target_h, target_w):
        """
        Args:
            x: (B, C, H/4, W/4)
            target_h, target_w: H, W
        Returns:
            out: (B, N, hidden_size)
        """
        feat        = self.upsample_net(x) # -> (B, hidden_size, H, W)
        
        if feat.shape[-2:] != (target_h, target_w):
            print(f"{Fore.RED}Robustness check not pass!{Fore.RESET}")
        
        # Safe injection via zero convolution
        feat        = self.zero_conv(feat)
        
        # Pool into tokens that match the patch size of the main DiT
        feat_token  = F.avg_pool2d(feat, kernel_size=self.patch_size, stride=self.patch_size) # -> (B, C, H/p, W/p)
        
        # Flatten into token sequence
        out         = feat_token.flatten(2).transpose(1, 2) # -> (B, N, hidden_size)
        
        return out


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    Standard DiT Block.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, drop_path=0.0, **block_kwargs):
        super().__init__()
        self.norm1            = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn             = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2            = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim        = int(hidden_size * mlp_ratio)
        approx_gelu           = lambda: nn.GELU(approximate="tanh")
        self.mlp              = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.drop_path        = DropPath(drop_path) if drop_path > 0. else nn.Identity() # NOTE:we add stochastic depth regulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + self.drop_path(gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa)))
        x = x + self.drop_path(gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final       = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear           = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x            = modulate(self.norm_final(x), shift, scale)
        x            = self.linear(x)
        return x


class SHIFT(nn.Module):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        out_channels=None, # Allow explicit override
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
        drop_path_rate=0.0,
        **kwargs
    ):
        super().__init__()
        self.learn_sigma  = learn_sigma
        self.in_channels  = in_channels
        self.out_channels = out_channels if out_channels is not None else (in_channels * 2 if learn_sigma else in_channels)
        self.patch_size   = patch_size
        self.num_heads    = num_heads
        self.input_size   = input_size

        # 1. Input Embeddings
        self.x_embedder   = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder   = TimestepEmbedder(hidden_size)
        
        # 2. Learnable Upsampling Encoder
        self.y_embedder   = LearnableUpsampleEncoder(in_channels, hidden_size, patch_size)
        
        # 3. Position Embedding
        num_patches       = self.x_embedder.num_patches
        self.pos_embed    = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # 4. Transformer Blocks
        dpr               = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks       = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=dpr[i])
            for i in range(depth)
        ])
        
        # 5. Final Layer
        self.final_layer  = FinalLayer(hidden_size, patch_size, self.out_channels)
        
        self.initialize_weights()

    def initialize_weights(self):
        # 1. Basic Init
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # 2. Pos Embed
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # 3. Patch Embed
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # 4. Timestep Embedder
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # 5. AdaLN Zero Init (Standard DiT)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        
        # 6. Zero Conv Init is handled in the class itself

    def unpatchify(self, x):
        c       = self.out_channels
        p       = self.x_embedder.patch_size[0]
        h       = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x       = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x       = torch.einsum('nhwpqc->nchpwq', x)
        imgs    = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, timesteps, y, force_drop_ids=None):
        """
        Args:
        -----
            x: (B, C1, H, W) noise
            t: (B,) timestep
            y: (B, C1, H, W) condition (Low Res)
        """
        if x is None:
            # Calculate target resolution from conditioning y and patch_size
            # NOTE
            scale = 4
            target_h, target_w = y.shape[2] * scale, y.shape[3] * scale
            x = torch.zeros(y.shape[0], self.in_channels, target_h, target_w, device=y.device)
        else:
            # Get target resolution from input noise
            target_h, target_w = x.shape[2], x.shape[3]

        # If t is not provided (Regression Mode), initialize it as zeros
        if timesteps is None:
            timesteps = torch.zeros(y.shape[0], device=y.device)

        # embeddings
        x                   = self.x_embedder(x) + self.pos_embed    # (B, N, hidden_size)
        t_emb               = self.t_embedder(timesteps)             # (B, hidden_size)
        spatial_bias        = self.y_embedder(y, target_h, target_w) # (B, N, hidden_size)
        
        # additive injection
        x                   = x + spatial_bias
        
        # dit blocks
        for block in self.blocks:
            x               = block(x, t_emb)

        # output
        x                   = self.final_layer(x, t_emb)
        x                   = self.unpatchify(x)

        return x


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb

def SHIFT_S_8_sr(**kwargs):
    return SHIFT(**models_config["shift_s_8_sr"], **kwargs)

def SHIFT_S_8_hard(**kwargs):
    return SHIFT(**models_config["shift_s_8_hard"], **kwargs)


#################################################################################
#                       SHIFT v2: Pixel-Friendly Variant                        #
#################################################################################
# Key improvements vs. v1 (shift.py:73-128, shift.py:180-311):
# 1) patch_size: 8 -> 4. For 128x128 input we now have 32x32=1024 tokens
#    (vs 256 tokens before). Reconstruction granularity is 4x finer.
# 2) HighResConditionEncoder: replaces LearnableUpsampleEncoder.
#    - Upsamples LR conditioning to TARGET resolution using PixelShuffle.
#    - Patchifies the conditioning with the SAME patch_size as x_embedder so
#      tokens are spatially aligned (no avg_pool that destroys high freq).
# 3) Residual SR shortcut: model predicts a residual on top of bilinear-upsampled
#    LR (strong SR inductive bias). Disabled for stochastic FM mode by setting
#    `use_lr_shortcut=False`.
# 4) drop_path_rate kept low (0.0) for tiny-overfit smoke tests.

class HighResConditionEncoder(nn.Module):
    """
    Upsample LR conditioning to HR resolution via stacked PixelShuffle, then
    tokenize with the SAME patch_size as the main DiT, so each conditioning
    token is spatially aligned with each x token. No avg_pool, so high
    frequencies in the conditioning are preserved.
    """
    def __init__(self, in_channels, hidden_size, patch_size, num_upsample_steps=2):
        super().__init__()
        self.patch_size = patch_size
        layers = [nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), nn.SiLU()]
        for _ in range(num_upsample_steps):
            layers += [
                nn.Conv2d(64, 64 * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.SiLU(),
            ]
        layers              += [nn.Conv2d(64, hidden_size, kernel_size=3, padding=1)]
        self.upsample_net   = nn.Sequential(*layers)

        # Aligned patchify (no pooling)
        self.patchify       = nn.Conv2d(hidden_size, hidden_size, kernel_size=patch_size, stride=patch_size)
        # Zero-init for safe additive injection
        self.zero_proj      = nn.Conv2d(hidden_size, hidden_size, kernel_size=1)
        nn.init.zeros_(self.zero_proj.weight)
        nn.init.zeros_(self.zero_proj.bias)

    def forward(self, y, target_h, target_w):
        feat        = self.upsample_net(y)                              # (B, C, H, W)
        if feat.shape[-2:] != (target_h, target_w):
            feat    = F.interpolate(feat, size=(target_h, target_w), mode='bilinear', align_corners=False)
        feat        = self.patchify(feat)                               # (B, C, H/p, W/p)
        feat        = self.zero_proj(feat)
        return feat.flatten(2).transpose(1, 2)                   # (B, N, C)


class SHIFTv2(nn.Module):
    """
    Pixel-friendly DiT for periodic Kolmogorov-flow super-resolution.
    Drop-in compatible signature with SHIFT (forward(x, timesteps, y)).
    """
    def __init__(
        self,
        input_size=128,
        patch_size=4,
        in_channels=2,
        out_channels=2,
        hidden_size=256,
        depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        learn_sigma=False,
        drop_path_rate=0.0,
        cond_upsample_steps=2,
        use_lr_shortcut=True,
        **kwargs,
    ):
        super().__init__()
        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.patch_size     = patch_size
        self.input_size     = input_size
        self.use_lr_shortcut = use_lr_shortcut

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = HighResConditionEncoder(in_channels, hidden_size, patch_size, num_upsample_steps=cond_upsample_steps)

        # self.concat_proj = nn.Linear(2 * hidden_size, hidden_size)

        num_patches    = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        dpr            = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks    = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels)
        self._initialize_weights()

    def _initialize_weights(self):
        def _basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
        self.apply(_basic)

        pe = get_2d_sincos_pos_embed(self.pos_embed.shape[-1],
                                     int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pe).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.zeros_(self.x_embedder.proj.bias)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for blk in self.blocks:
            nn.init.zeros_(blk.adaLN_modulation[-1].weight)
            # nn.init.normal_(blk.adaLN_modulation[-1].weight, std=0.02)
            nn.init.zeros_(blk.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(x.shape[0], c, h * p, h * p)

    def forward(self, x, timesteps, y, **kwargs):
        if x is None:
            scale               = self.input_size // y.shape[-1]
            target_h, target_w  = y.shape[2] * scale, y.shape[3] * scale
            x                   = torch.zeros(y.shape[0], self.in_channels, target_h, target_w, device=y.device, dtype=y.dtype)
        target_h, target_w      = x.shape[2], x.shape[3]
        if timesteps is None:
            timesteps = torch.zeros(y.shape[0], device=y.device)

        # Bilinear LR shortcut (residual SR; only valid when out_channels==in_channels)
        lr_up       = None
        if self.use_lr_shortcut and self.out_channels == self.in_channels:
            lr_up   = F.interpolate(y, size=(target_h, target_w), mode='bilinear', align_corners=False)

        x_tok       = self.x_embedder(x) + self.pos_embed
        t_emb       = self.t_embedder(timesteps)
        y_tok       = self.y_embedder(y, target_h, target_w)
        # 1.add
        x_tok       = x_tok + y_tok
        # 2.concat
        # x_tok  = torch.cat([x_tok, y_tok], dim=-1)
        # x_tok  = self.concat_proj(x_tok)
        for blk in self.blocks:
            x_tok   = blk(x_tok, t_emb)
        x_tok       = self.final_layer(x_tok, t_emb)
        out         = self.unpatchify(x_tok)
        if lr_up is not None:
            out     = out + lr_up
        return out


def SHIFT_v2_B_4_sr(**kwargs):
    cfg = dict(models_config["shift_v2_b_4_sr"]); cfg.update(kwargs)
    return SHIFTv2(**cfg)
    # return SHIFTv2(**models_config["shift_v2_b_4_sr"], **kwargs)

def SHIFT_v2_B_4_hard(**kwargs):
    # return SHIFTv2(**models_config["shift_v2_b_4_hard"], **kwargs)
    cfg = dict(models_config["shift_v2_b_4_hard"]); cfg.update(kwargs)
    return SHIFTv2(**cfg)

models_config = {
    "shift_s_8_hard": {
        "input_size": 128,
        "in_channels": 2,
        "out_channels": 3, # [psi, u_mean, v_mean]
        #============================
        "depth": 12,
        "hidden_size": 384,
        "patch_size": 8,
        "num_heads": 6,
        #============================
        "mlp_ratio": 4.0,
        "class_dropout_prob": 0.1,
        "num_classes": 1000,
        "learn_sigma": False,
        "use_label_condition": False,
        "drop_path_rate": 0.1,
    },
    "shift_s_8_sr": {
        "input_size": 128,
        "in_channels": 2,
        "out_channels": 2, # [u_mean, v_mean]
        #============================
        "depth": 12,
        "hidden_size": 384,
        "patch_size": 8,
        "num_heads": 6,
        #============================
        "mlp_ratio": 4.0,
        "class_dropout_prob": 0.1,
        "num_classes": 1000,
        "learn_sigma": False,
        "use_label_condition": False,
        "drop_path_rate": 0.1,
    },

    # === SHIFT v2: pixel-friendly configs ===
    # Token count for 128x128 with patch=4: 32x32 = 1024 (4x finer than v1).
    # Use depth=8, hidden=256 to keep params/compute moderate.
    "shift_v2_b_4_sr": {
        "input_size": 128,
        "in_channels": 2,
        "out_channels": 2,           # regression: [u, v]
        #============================
        # "depth": 12,
        # "hidden_size": 384,
        # "patch_size": 8,
        # "num_heads": 6,
        "depth": 8,
        "hidden_size": 256,
        "patch_size": 4,
        "num_heads": 8,
        #============================
        "mlp_ratio": 4.0,
        "learn_sigma": False,
        "drop_path_rate": 0.0,
        "cond_upsample_steps": 2,    # LR(H/4) -> HR via two PixelShuffle x2
        "use_lr_shortcut": True,     # residual on top of bilinear-upsampled LR
    },
    "shift_v2_b_4_hard": {
        "input_size": 128,
        "in_channels": 2,
        "out_channels": 3,           # hard-constraint: [psi, u_mean, v_mean]
        #============================
        # "depth": 12,
        # "hidden_size": 384,
        # "patch_size": 8,
        # "num_heads": 6,
        "depth": 8,
        "hidden_size": 256,
        "patch_size": 4,
        "num_heads": 8,
        # "depth": 12,
        # "hidden_size": 384,
        # "patch_size": 4,
        # "num_heads": 8,
        #============================
        "mlp_ratio": 4.0,
        "learn_sigma": False,
        "drop_path_rate": 0.0,
        "cond_upsample_steps": 2,    # 
        "use_lr_shortcut": False,    # disabled because out!=in (psi vs uv)
    },
}