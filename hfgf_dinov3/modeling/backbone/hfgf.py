"""High-Frequency Guided Fusion (HFGF) block, Sec. 3.2 and Algorithm 2 of the paper."""

import torch
import torch.nn as nn


class SpatialConv(nn.Module):
    """Standard spatial convolution for local feature refinement."""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )

    def forward(self, x):
        return self.conv(x)


class HFGFBlock(nn.Module):
    """High-Frequency Guided Fusion at one pyramid scale.

    The scalar ``hf_gamma`` jointly controls gradient injection and its spatial
    gate.
    """

    def __init__(
        self,
        channels: int,
        prior_channels: int,
        edge_gamma: float = 0.3,
        hf_gamma_init: float = 0.0,
        enable_spatial: bool = True,
        enable_freq: bool = True,
    ):
        super().__init__()
        self.enable_spatial = bool(enable_spatial)
        self.enable_freq = bool(enable_freq)
        in_ch = channels + prior_channels + channels  # high + prior + hf_proj
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch, channels, kernel_size=1, bias=False),
            nn.SyncBatchNorm(channels),
            nn.GELU(),
        )
        # Depthwise context branches
        self.dw1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.pw1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.SyncBatchNorm(channels),
            nn.GELU(),
        )
        self.dw2 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels, bias=False)
        self.pw2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.SyncBatchNorm(channels),
            nn.GELU(),
        )
        self.defconv = nn.Sequential(
            SpatialConv(channels, channels, kernel_size=3, padding=1),
            nn.SyncBatchNorm(channels),
            nn.GELU(),
        )
        self.mix = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.SyncBatchNorm(channels),
            nn.GELU(),
        )

        # Spatial and prior gates
        self.edge_gamma = edge_gamma
        self.edge_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True)
        self.prior_proj = nn.Conv2d(prior_channels, channels, kernel_size=1, bias=False)

        self.hf_proj = nn.Conv2d(1, channels, kernel_size=1, bias=False)
        self.hf_gate_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True)
        self.hf_gamma = nn.Parameter(torch.tensor(float(hf_gamma_init)))

    def _freq_gate(self, x):
        """Checkpoint-compatible amplitude gate on the unshifted FFT grid."""
        m = x.mean(dim=1, keepdim=True)  # (B,1,H,W)
        fft = torch.fft.fft2(m, norm="ortho")
        amp = torch.abs(fft)
        _, _, H, W = amp.shape
        yy, xx = torch.meshgrid(
            torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing="ij"
        )
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
        r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        r_norm = r / (r.max() + 1e-6)
        low_mask = (r_norm <= 0.25).float()
        high_mask = (r_norm >= 0.5).float()
        low = (amp * low_mask).mean(dim=(2, 3), keepdim=True)
        high = (amp * high_mask).mean(dim=(2, 3), keepdim=True)
        score = high - low
        return torch.sigmoid(score)

    def forward(self, high: torch.Tensor, prior: torch.Tensor, hf: torch.Tensor) -> torch.Tensor:
        if (hf is None) or (not self.enable_spatial):
            hf = torch.zeros_like(prior[:, :1])
        hf_proj = self.hf_proj(hf) * self.hf_gamma if self.enable_spatial else torch.zeros_like(high)

        x = torch.cat([high, prior, hf_proj], dim=1)
        x = self.fuse(x)

        a = self.pw1(self.dw1(x))
        b = self.pw2(self.dw2(x))
        c = self.defconv(x)
        x = self.mix(torch.cat([a, b, c], dim=1))

        # Frequency gate
        if self.enable_freq:
            g_freq = self._freq_gate(x)
            x = x * (1.0 + g_freq)

        # Prior edge gate
        g_edge = torch.sigmoid(self.edge_conv(prior.mean(dim=1, keepdim=True)))
        x = x * (1.0 + self.edge_gamma * g_edge)

        # HF spatial gate (shares hf_gamma)
        if self.enable_spatial:
            g_hf = torch.sigmoid(self.hf_gate_conv(hf))
            x = x * (1.0 + self.hf_gamma * g_hf)

        prior_red = self.prior_proj(prior)
        return x + high + 0.3 * prior_red

