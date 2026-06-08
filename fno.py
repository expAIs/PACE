
"""
2D Fourier Neural Operator for periodic Kolmogorov-flow super-resolution.

References:
    Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A.,
    Anandkumar, A. "Fourier Neural Operator for Parametric Partial Differential
    Equations." ICLR 2021.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """
    Truncated 2D spectral convolution. 
    Multiplies the lowest (modes1, modes2)
    Fourier coefficients with a learnable complex weight. 
    Periodic by construction.
    """
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes1       = modes1
        self.modes2       = modes2

        scale = 1.0 / (in_channels * out_channels)
        # We split complex weights into real and imaginary parts to support DDP/NCCL
        self.weights1_real = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2)
        )
        self.weights1_imag = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2)
        )
        self.weights2_real = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2)
        )
        self.weights2_imag = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2)
        )

    @property
    def weights1(self):
        return torch.complex(self.weights1_real, self.weights1_imag)

    @property
    def weights2(self):
        return torch.complex(self.weights2_real, self.weights2_imag)

    @staticmethod
    def _compl_mul2d(x, w):
        # x: (B, in_c, M1, M2), w: (in_c, out_c, M1, M2) -> (B, out_c, M1, M2)
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x):
        B, _, H, W = x.shape
        x_ft  = torch.fft.rfft2(x, norm="ortho")                # (B, C, H, W//2+1)
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)

        m1 = min(self.modes1, H // 2)
        m2 = min(self.modes2, W // 2 + 1)

        out_ft[:, :, :m1, :m2] = self._compl_mul2d(
            x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2]
        )
        out_ft[:, :, -m1:, :m2] = self._compl_mul2d(
            x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2]
        )
        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


class FNOBlock(nn.Module):
    """
    Standard FNO block: spectral_conv + pointwise(1x1 conv) + GELU,
    optionally with FiLM-style modulation by a timestep embedding.
    """
    def __init__(self, channels: int, modes1: int, modes2: int, time_dim: int = 0):
        super().__init__()
        self.spectral       = SpectralConv2d(channels, channels, modes1, modes2)
        self.pointwise      = nn.Conv2d(channels, channels, kernel_size=1)
        self.act            = nn.GELU()
        self.time_dim       = time_dim
        if time_dim > 0:
            self.time_proj  = nn.Linear(time_dim, 2 * channels)
            nn.init.zeros_(self.time_proj.weight)
            nn.init.zeros_(self.time_proj.bias)

    def forward(self, x, t_emb=None):
        h = self.spectral(x) + self.pointwise(x)
        if self.time_dim > 0 and t_emb is not None:
            scale, shift = self.time_proj(t_emb).chunk(2, dim=-1)
            scale = scale.unsqueeze(-1).unsqueeze(-1)
            shift = shift.unsqueeze(-1).unsqueeze(-1)
            h = h * (1 + scale) + shift
        return self.act(h)


def _sinusoidal_time_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0):
    half    = dim // 2
    freqs   = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args    = t.float()[:, None] * freqs[None]
    emb     = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class FNO2d(nn.Module):
    """
    Backbone:
        Lift (1x1 conv): in_channels   -> width
        N x FNOBlock  : width          -> width
        Project       : width          -> 128 -> out_channels (two 1x1 convs + GELU)

    Conditioning on low-res input is done by bilinear-upsampling y to the target
    resolution and concatenating along the channel dimension before the lift.
    """
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        width: int = 64,
        depth: int = 4,
        modes: int = 32,
        time_dim: int = 128,
        use_lr_shortcut: bool = True,
        cond_in_channels: int = 2,
    ):
        super().__init__()
        self.in_channels      = in_channels
        self.cond_in_channels = cond_in_channels
        self.out_channels     = out_channels
        self.use_lr_shortcut  = use_lr_shortcut
        self.time_dim         = time_dim

        # Time/timestep embedding (used only when timesteps is not None).
        self.t_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Lift: x and upsampled y are concatenated -> 2 * cond_in_channels channels (or in+cond).
        lift_in = in_channels + cond_in_channels + 2  # +2 for grid (x, y) coords
        self.lift = nn.Conv2d(lift_in, width, kernel_size=1)

        self.blocks = nn.ModuleList([
            FNOBlock(width, modes, modes, time_dim=time_dim) for _ in range(depth)
        ])

        self.proj = nn.Sequential(
            nn.Conv2d(width, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(128, out_channels, kernel_size=1),
        )

        # Initialize the final projection close to zero so the residual shortcut
        # dominates at the start of training (huge inductive bias for SR).
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    @staticmethod
    def _grid(B: int, H: int, W: int, device, dtype):
        gx      = torch.linspace(0.0, 1.0, H, device=device, dtype=dtype)
        gy      = torch.linspace(0.0, 1.0, W, device=device, dtype=dtype)
        gx, gy  = torch.meshgrid(gx, gy, indexing='ij')
        grid    = torch.stack([gx, gy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        return grid

    def forward(self, x, timesteps=None, y=None, **kwargs):
        if y is None:
            raise ValueError("FNO2d requires low-res conditioning `y`.")

        # Determine target resolution.
        if x is not None:
            target_h, target_w  = x.shape[2], x.shape[3]
        else:
            # If no x is given (pure regression), infer from y.
            scale               = kwargs.get("scale", 4)
            target_h, target_w  = y.shape[2] * scale, y.shape[3] * scale
            x                   = torch.zeros(y.shape[0], self.in_channels, target_h, target_w, device=y.device, dtype=y.dtype)

        B           = x.shape[0]
        y_up        = F.interpolate(y, size=(target_h, target_w), mode='bilinear', align_corners=False)
        grid        = self._grid(B, target_h, target_w, x.device, x.dtype)

        h           = torch.cat([x, y_up, grid], dim=1)
        h           = self.lift(h)

        # Timestep embedding
        if timesteps is None:
            t_emb   = None
        else:
            t_emb   = _sinusoidal_time_embedding(timesteps, self.time_dim)
            t_emb   = self.t_mlp(t_emb)

        for blk in self.blocks:
            h       = blk(h, t_emb)

        out         = self.proj(h)

        # Residual SR shortcut: only valid when out_channels matches in_channels.
        if self.use_lr_shortcut and self.out_channels == self.cond_in_channels:
            out     = out + y_up
        return out


# =============================================================================
# Factory: regression and hard-constraint variants
# =============================================================================

FNO_CONFIGS = {
    "fno_b_sr": dict(
        in_channels=2, out_channels=2, cond_in_channels=2,
        width=64, depth=4, modes=32, time_dim=128,
        use_lr_shortcut=True,
    ),
    "fno_b_hard": dict(
        in_channels=2, out_channels=3, cond_in_channels=2,
        width=64, depth=4, modes=32, time_dim=128,
        use_lr_shortcut=False,
    )
}

def FNO_B_SR(**kwargs):
    cfg = dict(FNO_CONFIGS["fno_b_sr"]); cfg.update(kwargs)
    return FNO2d(**cfg)

def FNO_B_HARD(**kwargs):
    cfg = dict(FNO_CONFIGS["fno_b_hard"]); cfg.update(kwargs)
    return FNO2d(**cfg)
