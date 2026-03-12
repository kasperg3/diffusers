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

"""S3Diff fine-tuning script for one-step image super-resolution.

Fine-tunes the S3Diff LoRA adapters (VAE + UNet) and the degradation-guidance
MLPs on paired LR/HR image data. The DEResNet is kept frozen.

Example usage (requires a CUDA GPU, ≥16 GB VRAM recommended):

    # Single-GPU
    python train.py \\
        --lr_data_dir /path/to/lr_images \\
        --hr_data_dir /path/to/hr_images \\
        --output_dir ./output \\
        --num_train_epochs 5 \\
        --train_batch_size 2 \\
        --learning_rate 5e-5 \\
        --dtype fp16

    # Multi-GPU with accelerate
    accelerate launch --num_processes 4 train.py \\
        --lr_data_dir /path/to/lr_images \\
        --hr_data_dir /path/to/hr_images \\
        --output_dir ./output \\
        --num_train_epochs 5 \\
        --train_batch_size 2 \\
        --learning_rate 5e-5 \\
        --dtype fp16
"""

import argparse
import os

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune S3Diff on paired LR/HR image data")
    parser.add_argument(
        "--lr_data_dir",
        type=str,
        required=True,
        help="Directory containing low-resolution training images.",
    )
    parser.add_argument(
        "--hr_data_dir",
        type=str,
        required=True,
        help="Directory containing high-resolution target images (must match LR filenames).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./s3diff-finetuned",
        help="Output directory for checkpoints and logs.",
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
        help="Path or HuggingFace Hub repo ID for the S3Diff adapter weights (starting checkpoint).",
    )
    parser.add_argument(
        "--de_net_path",
        type=str,
        default=None,
        help=(
            "Path or Hub repo ID for the DEResNet weights. "
            "If not provided, uses the same repo as --s3diff_path."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Target HR training resolution. LR images are resized to resolution // scale_factor.",
    )
    parser.add_argument(
        "--scale_factor",
        type=int,
        default=4,
        help="Super-resolution scale factor. Default: 4.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=2,
        help="Per-device training batch size.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Adam beta1 parameter.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="Adam beta2 parameter.",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
        help="Adam weight decay.",
    )
    parser.add_argument(
        "--lambda_l2",
        type=float,
        default=1.0,
        help="Weight for the L2 pixel reconstruction loss.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a checkpoint every N training steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Mixed precision dtype. Default: fp32.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of dataloader worker processes.",
    )
    parser.add_argument(
        "--pos_prompt",
        type=str,
        default="",
        help="Positive text prompt used during training forward pass.",
    )
    parser.add_argument(
        "--neg_prompt",
        type=str,
        default="",
        help="Negative text prompt (for online negative sample generation).",
    )
    return parser.parse_args()


class PairedSRDataset(Dataset):
    """Simple paired LR/HR dataset for super-resolution fine-tuning.

    Expects the LR and HR directories to contain images with matching filenames.
    HR images are center-cropped to ``hr_size``; LR images are derived by
    downsampling the HR crop by ``scale_factor``.
    """

    SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

    def __init__(self, lr_dir: str, hr_dir: str, hr_size: int = 512, scale_factor: int = 4):
        self.hr_size = hr_size
        self.lr_size = hr_size // scale_factor
        self.scale_factor = scale_factor

        self.hr_paths = sorted(
            os.path.join(hr_dir, f)
            for f in os.listdir(hr_dir)
            if os.path.splitext(f)[1].lower() in self.SUPPORTED_EXTS
        )
        lr_files = {os.path.splitext(f)[0] for f in os.listdir(lr_dir)}

        # Keep only HR images that have a matching LR file
        self.pairs = []
        for hr_path in self.hr_paths:
            stem = os.path.splitext(os.path.basename(hr_path))[0]
            if stem in lr_files:
                lr_candidates = [
                    os.path.join(lr_dir, stem + ext) for ext in self.SUPPORTED_EXTS if
                    os.path.exists(os.path.join(lr_dir, stem + ext))
                ]
                if lr_candidates:
                    self.pairs.append((lr_candidates[0], hr_path))

        if not self.pairs:
            raise ValueError(
                f"No matching LR/HR pairs found in '{lr_dir}' and '{hr_dir}'. "
                "Make sure image filenames (excluding extension) match."
            )

        self.hr_transform = transforms.Compose([
            transforms.Resize(hr_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(hr_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.lr_transform = transforms.Compose([
            transforms.Resize(self.lr_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.lr_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        from PIL import Image

        lr_path, hr_path = self.pairs[idx]
        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")
        return {"lr": self.lr_transform(lr_img), "hr": self.hr_transform(hr_img)}


def main():
    args = parse_args()

    mixed_precision_map = {"fp32": "no", "fp16": "fp16", "bf16": "bf16"}
    accelerator = Accelerator(
        mixed_precision=mixed_precision_map[args.dtype],
        gradient_accumulation_steps=1,
    )

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    # ---- Load pipeline -----------------------------------------------
    from diffusers import S3DiffPipeline

    accelerator.print(f"Loading base model from '{args.sd_turbo_path}' ...")
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    torch_dtype = dtype_map[args.dtype]

    pipe = S3DiffPipeline.from_pretrained(args.sd_turbo_path, torch_dtype=torch_dtype)

    # ---- Load S3Diff adapter weights (starting point) ----------------
    accelerator.print(f"Loading S3Diff weights from '{args.s3diff_path}' ...")
    pipe.load_s3diff_weights(args.s3diff_path)

    # ---- Load frozen DEResNet ----------------------------------------
    de_net_repo = args.de_net_path if args.de_net_path is not None else args.s3diff_path
    accelerator.print(f"Loading DEResNet weights from '{de_net_repo}' ...")
    try:
        pipe.load_de_net_weights(de_net_repo)
        pipe.de_net.requires_grad_(False)
        pipe.de_net.eval()
    except Exception as exc:
        accelerator.print(f"Warning: could not load DEResNet weights ({exc}). Using zero scores.")

    # ---- Freeze base model, unfreeze LoRA + adapter ------------------
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    # Un-freeze LoRA parameters
    for name, param in pipe.vae.named_parameters():
        if "lora" in name:
            param.requires_grad_(True)
    for name, param in pipe.unet.named_parameters():
        if "lora" in name:
            param.requires_grad_(True)
    # Un-freeze S3DiffAdapter
    pipe.s3diff_adapter.requires_grad_(True)

    trainable_params = [p for p in pipe.parameters() if p.requires_grad]
    accelerator.print(
        f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}"
    )

    # ---- Optimizer ---------------------------------------------------
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
    )

    # ---- Dataset & DataLoader ----------------------------------------
    accelerator.print("Loading dataset ...")
    dataset = PairedSRDataset(
        lr_dir=args.lr_data_dir,
        hr_dir=args.hr_data_dir,
        hr_size=args.resolution,
        scale_factor=args.scale_factor,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    accelerator.print(f"Dataset size: {len(dataset)} pairs")

    # ---- Prepare with Accelerator ------------------------------------
    pipe.vae, pipe.unet, pipe.s3diff_adapter, optimizer, dataloader = accelerator.prepare(
        pipe.vae, pipe.unet, pipe.s3diff_adapter, optimizer, dataloader
    )
    if pipe.de_net is not None:
        pipe.de_net = accelerator.prepare(pipe.de_net)

    pipe.text_encoder = pipe.text_encoder.to(accelerator.device)
    pipe.scheduler.set_timesteps(1, device=accelerator.device)
    timesteps = pipe.scheduler.timesteps  # [999]

    # ---- Training loop -----------------------------------------------
    global_step = 0

    for epoch in range(args.num_train_epochs):
        pipe.unet.train()
        pipe.vae.train()
        pipe.s3diff_adapter.train()

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{args.num_train_epochs}",
            disable=not accelerator.is_local_main_process,
        )

        for batch in progress_bar:
            lr_imgs = batch["lr"]  # (B, 3, lr_size, lr_size) in [0, 1]
            hr_imgs = batch["hr"]  # (B, 3, hr_size, hr_size) in [-1, 1]
            B = lr_imgs.shape[0]

            # Upscale LR to HR resolution
            lr_upscaled = F.interpolate(
                lr_imgs,
                size=(args.resolution, args.resolution),
                mode="bilinear",
                align_corners=False,
            ).contiguous()
            lr_norm = (lr_upscaled * 2.0 - 1.0).clamp(-1.0, 1.0)

            # Estimate degradation (frozen)
            with torch.no_grad():
                if pipe.de_net is not None:
                    deg_score = pipe.de_net(lr_imgs)
                else:
                    deg_score = torch.zeros(B, 2, device=accelerator.device, dtype=lr_imgs.dtype)

            # Set LoRA modulation
            pipe._set_lora_modulation(deg_score)

            # Encode to latent
            lq_latent = pipe.vae.encode(lr_norm).latent_dist.sample() * pipe.vae.config.scaling_factor

            # Encode prompts
            pos_prompts = [args.pos_prompt] * B
            pos_embeds = pipe._encode_prompt(pos_prompts, accelerator.device)

            t_batch = timesteps.expand(B)

            # Forward through UNet
            model_pred = pipe.unet(lq_latent, t_batch, encoder_hidden_states=pos_embeds).sample

            # Denoise step
            x_denoised = pipe.scheduler.step(model_pred, timesteps[0], lq_latent).prev_sample

            # Decode
            pred_image = pipe.vae.decode(x_denoised / pipe.vae.config.scaling_factor).sample

            # L2 loss in pixel space (HR is in [-1, 1])
            loss = F.mse_loss(pred_image.float(), hr_imgs.float()) * args.lambda_l2

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

            if accelerator.is_main_process:
                progress_bar.set_postfix(loss=loss.detach().item(), step=global_step)

                if global_step % args.checkpointing_steps == 0:
                    ckpt_path = os.path.join(
                        args.output_dir, "checkpoints", f"step_{global_step}.pt"
                    )
                    # Save adapter state (LoRA + S3DiffAdapter)
                    save_state = {
                        "s3diff_adapter": accelerator.unwrap_model(pipe.s3diff_adapter).state_dict(),
                        "global_step": global_step,
                    }
                    torch.save(save_state, ckpt_path)
                    accelerator.print(f"Saved checkpoint to {ckpt_path}")

        accelerator.print(f"Epoch {epoch + 1} complete. Loss: {loss.detach().item():.4f}")

    accelerator.print(f"Training complete. {global_step} steps total.")
    accelerator.wait_for_everyone()

    # Save final checkpoint
    if accelerator.is_main_process:
        final_path = os.path.join(args.output_dir, "s3diff_finetuned.pt")
        save_state = {
            "s3diff_adapter": accelerator.unwrap_model(pipe.s3diff_adapter).state_dict(),
            "global_step": global_step,
        }
        torch.save(save_state, final_path)
        accelerator.print(f"Final checkpoint saved to {final_path}")
        accelerator.print("\n✅ To run inference with your fine-tuned weights:")
        accelerator.print(
            f"   python inference.py --input <img> --output ./output "
            f"--s3diff_path {args.output_dir} --s3diff_filename s3diff_finetuned.pt"
        )


if __name__ == "__main__":
    main()
