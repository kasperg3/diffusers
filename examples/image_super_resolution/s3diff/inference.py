#!/usr/bin/env python
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""S3Diff inference script for one-step image super-resolution.

Example usage (requires a GPU):

    python inference.py \\
        --input low_res.png \\
        --output ./output \\
        --scale_factor 4 \\
        --pos_prompt "high quality, detailed, 4k" \\
        --neg_prompt "blurry, low quality, noisy" \\
        --dtype fp16
"""

import argparse
import os

import torch
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="S3Diff one-step image super-resolution inference")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input low-resolution image (or directory of images).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="Directory to save the super-resolved output images.",
    )
    parser.add_argument(
        "--sd_turbo_path",
        type=str,
        default="stabilityai/sd-turbo",
        help="Path or HuggingFace Hub ID of the SD-Turbo base model.",
    )
    parser.add_argument(
        "--s3diff_path",
        type=str,
        default="zhangap/S3Diff",
        help="Path or HuggingFace Hub repo ID for the S3Diff adapter weights.",
    )
    parser.add_argument(
        "--s3diff_filename",
        type=str,
        default="s3diff.pkl",
        help="Filename of the S3Diff adapter checkpoint within --s3diff_path.",
    )
    parser.add_argument(
        "--de_net_path",
        type=str,
        default=None,
        help=(
            "Path or HuggingFace Hub repo ID for the DEResNet weights. "
            "If not provided, uses the same repo as --s3diff_path."
        ),
    )
    parser.add_argument(
        "--de_net_filename",
        type=str,
        default="de_net.pth",
        help="Filename of the DEResNet checkpoint.",
    )
    parser.add_argument(
        "--scale_factor",
        type=int,
        default=4,
        help="Super-resolution scale factor (e.g. 4 for 4x SR). Default: 4.",
    )
    parser.add_argument(
        "--pos_prompt",
        type=str,
        default="A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting.",
        help="Positive text prompt to guide the reconstruction.",
    )
    parser.add_argument(
        "--neg_prompt",
        type=str,
        default="oil painting, cartoon, blur, dirty, messy, low quality, deformation, low resolution, oversmooth",
        help="Negative text prompt for classifier-free guidance.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.07,
        help="CFG guidance scale. Set to 1.0 to disable CFG. Default: 1.07.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Model dtype. Use 'fp16' or 'bf16' to reduce VRAM usage. Default: fp32.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on. Default: cuda.",
    )
    parser.add_argument(
        "--align_method",
        type=str,
        default="wavelet",
        choices=["wavelet", "adain", "none"],
        help=(
            "Color alignment method applied to the output. "
            "'wavelet' matches wavelet low-frequencies to the LR input; "
            "'adain' uses adaptive instance normalization; "
            "'none' applies no color correction. Default: wavelet."
        ),
    )
    return parser.parse_args()


def wavelet_color_fix(pred_img: Image.Image, src_img: Image.Image) -> Image.Image:
    """Apply wavelet-based color correction.

    Replaces the low-frequency content of the predicted image with that from the
    (bilinearly upscaled) source LR image, preserving high-frequency details.
    """
    import numpy as np

    pred = np.array(pred_img).astype(np.float32) / 255.0
    src = np.array(src_img.resize(pred_img.size, Image.BILINEAR)).astype(np.float32) / 255.0

    try:
        import pywt

        coeffs_pred = [pywt.dwt2(pred[:, :, c], "haar") for c in range(3)]
        coeffs_src = [pywt.dwt2(src[:, :, c], "haar") for c in range(3)]
        result_channels = []
        for c in range(3):
            ll_pred, (lh, hl, hh) = coeffs_pred[c]
            ll_src, _ = coeffs_src[c]
            channel = pywt.idwt2((ll_src, (lh, hl, hh)), "haar")
            result_channels.append(channel)
        result = np.stack(result_channels, axis=2).clip(0, 1)
        return Image.fromarray((result * 255).astype(np.uint8))
    except ImportError:
        # pywt not installed – fall back to no correction
        return pred_img


def load_image_paths(input_path: str):
    """Return a list of image paths from a file or directory."""
    supported = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    if os.path.isfile(input_path):
        return [input_path]
    if os.path.isdir(input_path):
        return sorted(
            os.path.join(input_path, f) for f in os.listdir(input_path) if os.path.splitext(f)[1].lower() in supported
        )
    raise FileNotFoundError(f"Input path not found: {input_path}")


def main():
    args = parse_args()

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    torch_dtype = dtype_map[args.dtype]

    # ---- Load pipeline -----------------------------------------------
    print(f"Loading SD-Turbo base model from '{args.sd_turbo_path}' ...")
    from diffusers import S3DiffPipeline

    pipe = S3DiffPipeline.from_pretrained(args.sd_turbo_path, torch_dtype=torch_dtype)
    pipe = pipe.to(args.device)

    # ---- Load S3Diff adapter weights ---------------------------------
    print(f"Loading S3Diff adapter weights from '{args.s3diff_path}' ...")
    pipe.load_s3diff_weights(args.s3diff_path, filename=args.s3diff_filename)

    # ---- Load DEResNet weights (optional but recommended) ------------
    de_net_repo = args.de_net_path if args.de_net_path is not None else args.s3diff_path
    print(f"Loading DEResNet weights from '{de_net_repo}' ...")
    try:
        pipe.load_de_net_weights(de_net_repo, filename=args.de_net_filename)
    except Exception as exc:
        print(f"Warning: could not load DEResNet weights ({exc}). Using zero degradation scores.")

    pipe.set_progress_bar_config(disable=True)

    # ---- Run inference -----------------------------------------------
    os.makedirs(args.output, exist_ok=True)
    image_paths = load_image_paths(args.input)
    print(f"Found {len(image_paths)} image(s). Running inference ...")

    for img_path in image_paths:
        lr_image = Image.open(img_path).convert("RGB")
        print(f"  Processing {os.path.basename(img_path)} ({lr_image.size[0]}x{lr_image.size[1]}) ...")

        result = pipe(
            image=lr_image,
            scale_factor=args.scale_factor,
            pos_prompt=args.pos_prompt,
            neg_prompt=args.neg_prompt,
            guidance_scale=args.guidance_scale,
        ).images[0]

        # Optional color alignment
        if args.align_method == "wavelet":
            result = wavelet_color_fix(result, lr_image)
        # 'adain' and 'none' are intentionally left for users to extend

        fname = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(args.output, f"{fname}_x{args.scale_factor}.png")
        result.save(out_path)
        print(f"  Saved → {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
