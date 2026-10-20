<div align="center">

<h1>HFGF-DINOv3</h1>

<p><strong>High-Frequency Guided Fusion and phase-aware loss<br>
for camouflaged object detection</strong></p>

<p>Jisang Lee · Hyoungwon Seo · Eunkyung Kim · Haneol Jang</p>

<p><em>Pattern Recognition · Volume 183 · 115066 · 2027</em></p>

<p>
  <a href="https://doi.org/10.1016/j.patcog.2026.115066"><img src="https://img.shields.io/badge/Paper-Pattern%20Recognition-2455A4?style=flat-square" alt="Paper: Pattern Recognition"></a>
  <img src="https://img.shields.io/badge/Python-3.8-3776AB?style=flat-square" alt="Python 3.8">
  <img src="https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=flat-square" alt="PyTorch 2.1">
</p>

<p>
  <a href="https://doi.org/10.1016/j.patcog.2026.115066">Paper</a> ·
  <a href="#results">Results</a> ·
  <a href="#installation">Installation</a> ·
  <a href="#datasets">Datasets</a> ·
  <a href="#training">Training</a> ·
  <a href="#evaluation">Evaluation</a> ·
  <a href="#inference">Inference</a> ·
  <a href="#citation">Citation</a>
</p>

</div>

PyTorch implementation of HFGF-DINOv3. The model combines a DINOv3 ViT-H+/16 backbone with ViT-Split, high-frequency guided fusion, and the boundary-aware high-frequency phase loss (BaP-HF).

## Results

**Results reported in the paper**, as mean ± standard deviation across seeds 42, 84, and 126.

| Dataset | Images | S-measure ↑ | E-measure ↑ | Weighted F-measure ↑ | MAE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| CAMO | 250 | 0.920 ± 0.000 | 0.969 ± 0.001 | 0.917 ± 0.001 | 0.021 ± 0.001 |
| COD10K | 2,026 | 0.916 ± 0.001 | 0.976 ± 0.000 | 0.894 ± 0.000 | 0.012 ± 0.000 |
| NC4K | 4,121 | 0.917 ± 0.000 | 0.966 ± 0.001 | 0.910 ± 0.001 | 0.020 ± 0.001 |
| CHAMELEON | 76 | 0.927 ± 0.001 | 0.977 ± 0.001 | 0.921 ± 0.002 | 0.014 ± 0.001 |

## Installation

Tested on Linux with 3 × NVIDIA RTX A6000 (48 GB each).

| Python | PyTorch | torchvision | CUDA |
| :---: | :---: | :---: | :---: |
| 3.8.20 | 2.1.1 | 0.16.1 | 11.8 |

### Environment

Run these commands from the repository root:

```bash
conda create -n hfgf-dinov3 python=3.8.20 pip=24.0 -y
conda activate hfgf-dinov3
python -m pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

### Build operators

Install Detectron2 and the multi-scale deformable attention operator. Building requires the CUDA 11.8 toolkit (`nvcc`) and a compatible C++ compiler.

```bash
git clone https://github.com/facebookresearch/detectron2.git third_party/detectron2
git -C third_party/detectron2 checkout 525208d9d6b36c72500183470fd181848b822468
CUDA_VISIBLE_DEVICES=0,1,2 python -m pip install --no-build-isolation -e third_party/detectron2

cd hfgf_dinov3/modeling/pixel_decoder/ops
CUDA_VISIBLE_DEVICES=0,1,2 python -m pip install --no-build-isolation .
cd ../../../..
```

The DINOv3 backbone runtime is included in `dinov3/`.

<details>
<summary>Optional: run the CPU tests</summary>

```bash
python -m unittest discover -s tests -v
```

</details>

## Datasets

Download the datasets from their official sources:

| Dataset | Training images | Test images |
| --- | ---: | ---: |
| [CAMO](https://sites.google.com/view/ltnghia/research/camo) | 1,000 | 250 |
| [COD10K](https://github.com/DengPingFan/SINet) | 3,040 | 2,026 |
| [NC4K](https://github.com/JingZhang617/COD-Rank-Localize-and-Segment) | — | 4,121 |
| [CHAMELEON](https://www.polsl.pl/rau6/datasets/) | — | 76 |

Arrange the image and mask files as follows:

```text
datasets/
├── CAMO-V.1.0-CVIU2019/
│   ├── Images/Train/
│   ├── Images/Test/
│   └── GT/
├── COD10K-v3/
│   ├── Train/{Image,GT_Object}/
│   └── Test/{Image,GT_Object}/
├── NC4K/{Imgs,GT}/
└── CHAMELEON/{Imgs,GT}/
```

Prepare the annotations and set the dataset root:

```bash
python tools/prepare_datasets.py --root datasets
export DETECTRON2_DATASETS="$(pwd)/datasets"
```

Training combines the CAMO and COD10K training sets. NC4K and CHAMELEON are used only for evaluation.

## Training

Obtain the **DINOv3 ViT-H+/16 LVD-1689M** pretrained weights through the [official DINOv3 release](https://github.com/facebookresearch/dinov3#pretrained-models), and place the raw `.pth` file in `checkpoints/`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python train_net.py \
  --config-file configs/hfgf_dinov3.yaml --num-gpus 3 \
  MODEL.BACKBONE_WEIGHTS checkpoints/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth
```

The default configuration uses a total batch size of **9** for **17,778 updates**; the paper specifies **8** and **20,000**, respectively.

| Setting | Value |
| --- | --- |
| Optimizer | AdamW |
| Learning rate: backbone / head | `1e-5` / `1e-4` |
| Weight decay | `0.05` |
| Learning-rate schedule | Polynomial decay |
| Precision | Automatic mixed precision |
| Seed | `42` |

Checkpoints and logs are saved to `outputs/hfgf_dinov3/`. Add `--resume` with the same training configuration to continue an interrupted run.

## Evaluation

Evaluate the checkpoint produced by training on all four datasets:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python train_net.py \
  --config-file configs/hfgf_dinov3.yaml --num-gpus 3 --eval-only \
  MODEL.WEIGHTS outputs/hfgf_dinov3/model_final.pth
```

Test images are resized to a shortest side of 512 pixels, with the longest side capped at 2,048. Evaluation uses the model's merged foreground/background query mask restored to the original image resolution. E-measure is averaged over its threshold curve. The same metric implementation is used for saved prediction masks.

## Inference

Predict a single image or a directory of images:

```bash
CUDA_VISIBLE_DEVICES=0 python predict.py \
  --config-file configs/hfgf_dinov3.yaml \
  --checkpoint outputs/hfgf_dinov3/model_final.pth \
  --input path/to/images --output predictions
```

Predictions are binary PNG masks named after the input image stems. To evaluate saved CAMO predictions:

```bash
python tools/evaluate_predictions.py \
  --dataset cod_val --predictions predictions \
  --dataset-root datasets --output outputs/camo_metrics
```

Use `cod10k_val`, `nc4k`, or `chameleon` for the other test sets.

## Citation

```bibtex
@article{lee2027hfgfdinov3,
  title   = {{HFGF-DINOv3}: High-Frequency Guided Fusion and phase-aware loss for camouflaged object detection},
  author  = {Lee, Jisang and Seo, Hyoungwon and Kim, Eunkyung and Jang, Haneol},
  journal = {Pattern Recognition},
  volume  = {183},
  pages   = {115066},
  year    = {2027},
  doi     = {10.1016/j.patcog.2026.115066}
}
```

## Acknowledgements and licenses

This implementation builds on [FOCUS](https://github.com/geshang777/FOCUS), [DINOv3](https://github.com/facebookresearch/dinov3), [ViT-Split](https://github.com/JackYFL/ViT-Split), [Mask2Former](https://github.com/facebookresearch/Mask2Former), and [Detectron2](https://github.com/facebookresearch/detectron2). Third-party components retain their respective licenses; see [NOTICE](NOTICE) and [LICENSES](LICENSES). The bundled DINOv3 runtime and its pretrained weights are governed by the [DINOv3 License](dinov3/LICENSE.md).
