# Third-party notices

HFGF-DINOv3 is released under the [Apache License 2.0](LICENSE). The components below keep their original copyright notices and licenses; the source files carry the upstream headers.

| Component | Used in | Source | License |
|:--|:--|:--|:--|
| FOCUS | Segmentation framework, training/data pipeline, matcher, losses (base of `hfgf_dinov3/`) | [geshang777/FOCUS](https://github.com/geshang777/FOCUS) | Apache-2.0, [`LICENSE`](LICENSE) |
| Mask2Former | Meta-architecture, pixel/transformer decoder, mask head (through FOCUS) | [facebookresearch/Mask2Former](https://github.com/facebookresearch/Mask2Former) | MIT, [`licenses/Mask2Former-MIT.txt`](licenses/Mask2Former-MIT.txt) |
| DETR, Detectron2 | Matching, box and position utilities, trainer interfaces (Detectron2 is installed separately) | [facebookresearch/detr](https://github.com/facebookresearch/detr), [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2) | Apache-2.0, [`LICENSE`](LICENSE) |
| Deformable DETR | Multi-scale deformable attention operator in `hfgf_dinov3/modeling/pixel_decoder/ops/` | [fundamentalvision/Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR) | Apache-2.0, [`LICENSE`](LICENSE) |
| Explicit Visual Prompting | Saliency metrics in `hfgf_dinov3/evaluation/metrics.py` (through FOCUS) | [NiFangBaAGe/Explicit-Visual-Prompt](https://github.com/NiFangBaAGe/Explicit-Visual-Prompt) | BSD-3-Clause, [`licenses/Explicit-Visual-Prompt-BSD-3-Clause.txt`](licenses/Explicit-Visual-Prompt-BSD-3-Clause.txt) |
| DINOv3 | ViT backbone runtime in `third_party/dinov3/` and the pretrained weights | [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) | DINOv3 License, [`third_party/dinov3/LICENSE.md`](third_party/dinov3/LICENSE.md) |

**DINOv3 runtime.** `third_party/dinov3/` is based on DINOv3 commit `5aa7e93ab42ae3526291f6b4e3fcd22637e03326`, reduced to the ViT-H+/16 backbone runtime with compatibility changes for Python 3.8 and PyTorch 2.1. Pretraining, downstream tasks, other model variants and FP8 utilities were removed. The DINOv3 License Agreement governs this directory and the DINOv3 weights, independently of the Apache-2.0 license of the rest of the repository.

**ViT-Split.** The backbone design follows [ViT-Split](https://github.com/JackYFL/ViT-Split) ([arXiv:2506.03433](https://arxiv.org/abs/2506.03433)).

**Figures and data.** `assets/overview.png` and `assets/qualitative.png` are reproduced from the authors' article in *Pattern Recognition*. Benchmark datasets and pretrained weights obtained separately remain subject to their providers' terms.
