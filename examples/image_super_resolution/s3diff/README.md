# S3Diff: Degradation-Guided One-Step Image Super-Resolution

This directory contains inference and training scripts for [S3Diff](https://arxiv.org/abs/2409.17058), a one-step image super-resolution model built on SD-Turbo.

## Overview

S3Diff extends SD-Turbo with:
- A **Degradation Estimation Network (DEResNet)** that predicts per-image blur/noise scores
- **Degradation-guided LoRA** adapters on the VAE encoder and UNet that are dynamically re-weighted at inference

## GPU Test Scripts

The following scripts require a CUDA GPU. Run them in order:

### 1. Inference

```bash
# Basic inference (downloads weights automatically from HuggingFace)
python examples/image_super_resolution/s3diff/inference.py \
    --input path/to/low_res_image.png \
    --output path/to/output_dir \
    --scale_factor 4

# With custom positive/negative prompts
python examples/image_super_resolution/s3diff/inference.py \
    --input path/to/low_res_image.png \
    --output path/to/output_dir \
    --scale_factor 4 \
    --pos_prompt "high quality, detailed, 4k" \
    --neg_prompt "blurry, low quality, noisy"

# Half-precision (fp16) for lower VRAM usage
python examples/image_super_resolution/s3diff/inference.py \
    --input path/to/low_res_image.png \
    --output path/to/output_dir \
    --scale_factor 4 \
    --dtype fp16
```

### 2. Training

```bash
# Fine-tune S3Diff on your own paired LR/HR dataset
python examples/image_super_resolution/s3diff/train.py \
    --lr_data_dir path/to/low_resolution_images \
    --hr_data_dir path/to/high_resolution_images \
    --output_dir path/to/output_dir \
    --num_train_epochs 10 \
    --train_batch_size 4 \
    --learning_rate 5e-5
```

## Requirements

```bash
pip install diffusers transformers accelerate peft pillow torchvision
```

## Pre-trained Weights

Pre-trained weights are downloaded automatically from HuggingFace:
- **SD-Turbo base**: `stabilityai/sd-turbo`
- **S3Diff adapters**: `zhangap/S3Diff` (`s3diff.pkl`)
- **DEResNet**: `zhangap/S3Diff` (`de_net.pth`)

## Citation

```bibtex
@article{zhang2024s3diff,
  author    = {Aiping Zhang, Zongsheng Yue, Renjing Pei, Wenqi Ren, Xiaochun Cao},
  title     = {Degradation-Guided One-Step Image Super-Resolution with Diffusion Priors},
  journal   = {arXiv},
  year      = {2024},
}
```
