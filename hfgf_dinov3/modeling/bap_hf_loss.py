"""Boundary-aware Phase High-Frequency (BaP-HF) loss, Sec. 3.3 of the paper."""

import torch
import torch.fft as fft
import torch.nn.functional as F


def boundary_band(masks, kernel_size):
    """Soft boundary band B of Eq. (8): AvgPool(MaxPool(Y) - MinPool(Y)), clamped to [0, 1]."""
    k = max(1, int(kernel_size))
    pad = k // 2
    max_m = F.max_pool2d(masks, kernel_size=k, stride=1, padding=pad)
    min_m = -F.max_pool2d(-masks, kernel_size=k, stride=1, padding=pad)
    boundary = (max_m - min_m).clamp(min=0.0, max=1.0)
    return F.avg_pool2d(boundary, kernel_size=k, stride=1, padding=pad)


def bap_hf_loss(pred_logits, target_masks, highpass=0.25, band_kernel=3):
    """Cosine phase discrepancy in the high-frequency band, Eqs. (7)-(13).

    Args:
        pred_logits: matched mask logits, shape (N, 1, h, w).
        target_masks: matched ground-truth masks, shape (N, 1, H, W).
        highpass: normalized radial cutoff ``hp`` of the high-pass mask.
        band_kernel: kernel size ``k`` of the boundary band.

    The normalization is over frequency bins; all matched masks are summed.
    """
    # Eq. (7): upsample the logits to the ground-truth resolution.
    pred_prob = F.interpolate(
        pred_logits, size=target_masks.shape[-2:], mode="bilinear", align_corners=False
    ).sigmoid()

    with torch.no_grad():
        band = boundary_band(target_masks, band_kernel)

    if band.sum() < 1:
        return None

    # Eq. (9): restrict both maps to the boundary band.
    src_b = (pred_prob * band).float()
    tgt_b = (target_masks * band).float()

    # Eq. (10): centered complex spectra.
    f_pred = fft.fftshift(fft.fft2(src_b, dim=(-2, -1)), dim=(-2, -1))
    f_gt = fft.fftshift(fft.fft2(tgt_b, dim=(-2, -1)), dim=(-2, -1))

    eps = 1e-6
    # Eq. (12): high-pass radial mask on the normalized radius.
    _, _, H, W = f_gt.shape
    yy, xx = torch.meshgrid(
        torch.arange(H, device=f_gt.device),
        torch.arange(W, device=f_gt.device),
        indexing="ij",
    )
    yy = yy.float() - H / 2
    xx = xx.float() - W / 2
    rr = torch.sqrt(xx ** 2 + yy ** 2) / (0.5 * min(H, W) + eps)
    hp_mask = (rr >= float(highpass)).float().view(1, 1, H, W)

    # Eqs. (11) and (13): phase-only spectra and per-frequency cosine penalty.
    p_pred = f_pred / (torch.abs(f_pred) + eps)
    p_gt = f_gt / (torch.abs(f_gt) + eps)
    cos = torch.real(p_pred * torch.conj(p_gt)).clamp(min=-1.0, max=1.0)
    num = ((1.0 - cos) * hp_mask).sum()
    den = hp_mask.sum() + eps
    return num / den
