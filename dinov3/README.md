# DINOv3 runtime

This directory contains the ViT-H+/16 runtime used by HFGF-DINOv3, derived
from [Meta's DINOv3 implementation](https://github.com/facebookresearch/dinov3)
at commit `5aa7e93ab42ae3526291f6b4e3fcd22637e03326`.

It retains the backbone layers and local `dinov3_vith16plus` entrypoint.
Unused pretraining code, downstream applications, model variants, and FP8
utilities are omitted. Compatibility changes support Python 3.8 and PyTorch 2.1.
Follow the installation and checkpoint instructions in the parent README.

Copyright (c) Meta Platforms, Inc. and affiliates.
This code and the DINOv3 weights remain subject to the
[DINOv3 License Agreement](LICENSE.md).
