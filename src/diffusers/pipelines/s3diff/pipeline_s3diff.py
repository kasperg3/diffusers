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

"""S3Diff: Degradation-Guided One-Step Image Super-Resolution with Diffusion Priors.

Paper: https://arxiv.org/abs/2409.17058
Original code: https://github.com/ArcticHare105/S3Diff
"""

import math
from typing import Any, Callable, List, Optional, Union

import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer

from ...configuration_utils import ConfigMixin, register_to_config
from ...image_processor import PipelineImageInput, VaeImageProcessor
from ...models import AutoencoderKL, UNet2DConditionModel
from ...models.modeling_utils import ModelMixin
from ...schedulers import DDPMScheduler
from ...utils import is_peft_available, logging
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from .modeling_de_net import DEResNet


if is_peft_available():
    from peft import LoraConfig

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from PIL import Image
        >>> from diffusers import S3DiffPipeline

        >>> # Load base SD-Turbo model with S3Diff weights
        >>> pipe = S3DiffPipeline.from_pretrained(
        ...     "stabilityai/sd-turbo",
        ...     torch_dtype=torch.float16,
        ... )
        >>> pipe = pipe.to("cuda")

        >>> # Load S3Diff adapter weights
        >>> pipe.load_s3diff_weights("zhangap/S3Diff", filename="s3diff.pkl")

        >>> # Run super-resolution
        >>> lr_image = Image.open("low_res.png").convert("RGB")
        >>> result = pipe(
        ...     image=lr_image,
        ...     scale_factor=4,
        ...     pos_prompt="high quality, detailed",
        ...     neg_prompt="blurry, low quality",
        ... ).images[0]
        >>> result.save("high_res.png")
        ```
"""


def _s3diff_lora_forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    """Custom LoRA forward pass for S3Diff degradation-guided modulation.

    This replaces the standard PEFT LoRA forward with a version that applies
    a degradation-guided modulation matrix (``de_mod``) to the LoRA intermediate
    activations, enabling input-dependent (degradation-aware) LoRA scaling.

    The ``de_mod`` attribute must be set on the module before calling forward.
    It has shape ``(B, lora_rank, lora_rank)`` and is broadcast over the batch.
    """
    self._check_forward_args(x, *args, **kwargs)
    adapter_names = kwargs.pop("adapter_names", None)

    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        result = self.base_layer(x, *args, **kwargs)
    elif adapter_names is not None:
        result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
    elif self.merged:
        result = self.base_layer(x, *args, **kwargs)
    else:
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A:
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            x = x.to(lora_A.weight.dtype)

            if not self.use_dora[active_adapter]:
                _tmp = lora_A(dropout(x))
                if isinstance(lora_A, nn.Conv2d):
                    # _tmp: (B, rank, H, W); de_mod: (B, rank_in, rank_out)
                    _tmp = torch.einsum("bkhw,bkr->brhw", _tmp, self.de_mod)
                elif isinstance(lora_A, nn.Linear):
                    # _tmp: (..., rank); de_mod: (B, rank_in, rank_out)
                    _tmp = torch.einsum("...lk,bkr->...lr", _tmp, self.de_mod)
                else:
                    raise NotImplementedError(
                        f"Only Conv2d and Linear layers are supported in _s3diff_lora_forward, "
                        f"got {type(lora_A).__name__}."
                    )
                result = result + lora_B(_tmp) * scaling
            else:
                x = dropout(x)
                result = result + self._apply_dora(x, lora_A, lora_B, scaling, active_adapter)

        result = result.to(torch_result_dtype)

    return result


class S3DiffAdapter(ModelMixin, ConfigMixin):
    """Adapter module holding S3Diff-specific parameters.

    This module contains the degradation-guidance components:
    - Fourier feature weights ``W``
    - Degradation-to-embedding MLPs for VAE and UNet
    - Per-block embedding tables
    - Fuse MLPs that map (degradation_embed, block_embed) → LoRA modulation matrix

    These are separate from the base SD-Turbo model and get loaded from the
    ``s3diff.pkl`` checkpoint.

    Args:
        lora_rank_unet (int): LoRA rank for UNet. Default: 8.
        lora_rank_vae (int): LoRA rank for VAE encoder. Default: 4.
        num_embeddings (int): Dimension of Fourier features. Default: 64.
        block_embedding_dim (int): Dimension of block position embeddings. Default: 64.
        num_vae_blocks (int): Number of VAE encoder block embedding entries. Default: 6.
        num_unet_blocks (int): Number of UNet block embedding entries. Default: 10.
    """

    @register_to_config
    def __init__(
        self,
        lora_rank_unet: int = 8,
        lora_rank_vae: int = 4,
        num_embeddings: int = 64,
        block_embedding_dim: int = 64,
        num_vae_blocks: int = 6,
        num_unet_blocks: int = 10,
    ):
        super().__init__()

        self.register_buffer("W", torch.randn(num_embeddings))

        self.vae_de_mlp = nn.Sequential(nn.Linear(num_embeddings * 4, 256), nn.ReLU(True))
        self.unet_de_mlp = nn.Sequential(nn.Linear(num_embeddings * 4, 256), nn.ReLU(True))
        self.vae_block_mlp = nn.Sequential(nn.Linear(block_embedding_dim, 64), nn.ReLU(True))
        self.unet_block_mlp = nn.Sequential(nn.Linear(block_embedding_dim, 64), nn.ReLU(True))
        self.vae_fuse_mlp = nn.Linear(256 + 64, lora_rank_vae**2)
        self.unet_fuse_mlp = nn.Linear(256 + 64, lora_rank_unet**2)
        self.vae_block_embeddings = nn.Embedding(num_vae_blocks, block_embedding_dim)
        self.unet_block_embeddings = nn.Embedding(num_unet_blocks, block_embedding_dim)

    def compute_embeddings(self, deg_score: torch.Tensor):
        """Compute per-block LoRA modulation embeddings from degradation scores.

        Args:
            deg_score (torch.Tensor): Degradation scores of shape ``(B, 2)``.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: ``(vae_embeds, unet_embeds)`` with shapes
                ``(B, num_vae_blocks, rank_vae^2)`` and ``(B, num_unet_blocks, rank_unet^2)``.
        """
        # Fourier encoding: (B, 2) → (B, 4 * num_embeddings)
        deg_proj = deg_score[..., None] * self.W[None, None, :] * 2 * math.pi
        deg_proj = torch.cat([torch.sin(deg_proj), torch.cos(deg_proj)], dim=-1)
        deg_proj = torch.cat([deg_proj[:, 0], deg_proj[:, 1]], dim=-1)

        vae_de = self.vae_de_mlp(deg_proj)  # (B, 256)
        unet_de = self.unet_de_mlp(deg_proj)  # (B, 256)

        vae_blk = self.vae_block_mlp(self.vae_block_embeddings.weight)  # (6, 64)
        unet_blk = self.unet_block_mlp(self.unet_block_embeddings.weight)  # (10, 64)

        n_vae = vae_blk.shape[0]
        n_unet = unet_blk.shape[0]
        B = deg_score.shape[0]

        vae_embeds = self.vae_fuse_mlp(
            torch.cat([vae_de.unsqueeze(1).expand(-1, n_vae, -1), vae_blk.unsqueeze(0).expand(B, -1, -1)], dim=-1)
        )  # (B, n_vae, rank_vae^2)
        unet_embeds = self.unet_fuse_mlp(
            torch.cat(
                [unet_de.unsqueeze(1).expand(-1, n_unet, -1), unet_blk.unsqueeze(0).expand(B, -1, -1)], dim=-1
            )
        )  # (B, n_unet, rank_unet^2)

        return vae_embeds, unet_embeds


class S3DiffPipeline(DiffusionPipeline):
    """
    Pipeline for one-step image super-resolution using S3Diff.

    S3Diff enhances a pre-trained SD-Turbo model with:
    - Degradation-guided LoRA adapters on the VAE encoder and UNet
    - A degradation estimation network (DEResNet) to estimate blur/noise
    - One-step inference with optional classifier-free guidance

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder (CLIP ViT-L/14).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A ``CLIPTokenizer`` to tokenize text prompts.
        unet ([`UNet2DConditionModel`]):
            A ``UNet2DConditionModel`` to denoise the encoded image latents.
        scheduler ([`DDPMScheduler`]):
            A ``DDPMScheduler`` configured for one-step inference.
        s3diff_adapter ([`S3DiffAdapter`]):
            The S3Diff degradation-guidance adapter containing the MLPs and embeddings.
        de_net ([`~pipelines.s3diff.DEResNet`], *optional*):
            Degradation estimation network. When provided, degradation scores are
            estimated automatically from the input image.
    """

    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = ["de_net"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: DDPMScheduler,
        s3diff_adapter: S3DiffAdapter,
        de_net: Optional[DEResNet] = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            s3diff_adapter=s3diff_adapter,
            de_net=de_net,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        # Track which LoRA layers have been patched with S3Diff forward hooks
        self._vae_lora_layers: List[str] = []
        self._unet_lora_layers: List[str] = []
        self._lora_applied: bool = False

    # ------------------------------------------------------------------
    # LoRA setup helpers
    # ------------------------------------------------------------------

    def _apply_lora_to_vae(self, vae_lora_config: "LoraConfig") -> None:
        """Apply PEFT LoRA adapter to VAE encoder and install S3Diff forward hook."""
        self.vae.add_adapter(vae_lora_config, adapter_name="vae_skip")

        self._vae_lora_layers = []
        for name, module in self.vae.named_modules():
            if "base_layer" in name:
                self._vae_lora_layers.append(name[: -len(".base_layer")])

        for name, module in self.vae.named_modules():
            if name in self._vae_lora_layers:
                module.forward = _s3diff_lora_forward.__get__(module, module.__class__)

    def _apply_lora_to_unet(self, unet_lora_config: "LoraConfig") -> None:
        """Apply PEFT LoRA adapter to UNet and install S3Diff forward hook."""
        self.unet.add_adapter(unet_lora_config)

        self._unet_lora_layers = []
        for name, module in self.unet.named_modules():
            if "base_layer" in name:
                self._unet_lora_layers.append(name[: -len(".base_layer")])

        for name, module in self.unet.named_modules():
            if name in self._unet_lora_layers:
                module.forward = _s3diff_lora_forward.__get__(module, module.__class__)

    def load_s3diff_weights(self, pretrained_path: str, filename: str = "s3diff.pkl") -> None:
        """Load S3Diff adapter weights from a local file or HuggingFace Hub.

        This method:
        1. Loads the S3Diff checkpoint (a ``.pkl`` file).
        2. Applies PEFT LoRA adapters with the correct ranks to VAE and UNet.
        3. Restores the LoRA weights and all S3Diff-specific parameters (MLPs, embeddings).
        4. Installs the degradation-guided LoRA forward hooks.

        Args:
            pretrained_path (str): Path to the directory or HuggingFace repo containing
                the S3Diff checkpoint file.
            filename (str): Name of the checkpoint file. Default: ``"s3diff.pkl"``.
        """
        if not is_peft_available():
            raise ImportError(
                "PEFT is required to load S3Diff weights. "
                "Install it with: pip install peft"
            )

        import os

        if os.path.isfile(pretrained_path):
            ckpt_path = pretrained_path
        elif os.path.isdir(pretrained_path):
            ckpt_path = os.path.join(pretrained_path, filename)
        else:
            # Try HuggingFace Hub
            from huggingface_hub import hf_hub_download

            ckpt_path = hf_hub_download(repo_id=pretrained_path, filename=filename)

        logger.info(f"Loading S3Diff weights from {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu")

        # ---- VAE LoRA ------------------------------------------------
        rank_vae = sd["rank_vae"]
        vae_target_modules = sd["vae_lora_target_modules"]
        vae_lora_config = LoraConfig(
            r=rank_vae,
            init_lora_weights="gaussian",
            target_modules=vae_target_modules,
        )
        self._apply_lora_to_vae(vae_lora_config)

        # Restore VAE state
        _sd_vae = self.vae.state_dict()
        for k, v in sd["state_dict_vae"].items():
            _sd_vae[k] = v
        self.vae.load_state_dict(_sd_vae)

        # ---- UNet LoRA -----------------------------------------------
        rank_unet = sd["rank_unet"]
        unet_target_modules = sd["unet_lora_target_modules"]
        unet_lora_config = LoraConfig(
            r=rank_unet,
            init_lora_weights="gaussian",
            target_modules=unet_target_modules,
        )
        self._apply_lora_to_unet(unet_lora_config)

        # Restore UNet state
        _sd_unet = self.unet.state_dict()
        for k, v in sd["state_dict_unet"].items():
            _sd_unet[k] = v
        self.unet.load_state_dict(_sd_unet)

        # ---- S3Diff MLP / Embedding weights --------------------------
        self.s3diff_adapter.vae_de_mlp.load_state_dict(sd["state_dict_vae_de_mlp"])
        self.s3diff_adapter.unet_de_mlp.load_state_dict(sd["state_dict_unet_de_mlp"])
        self.s3diff_adapter.vae_block_mlp.load_state_dict(sd["state_dict_vae_block_mlp"])
        self.s3diff_adapter.unet_block_mlp.load_state_dict(sd["state_dict_unet_block_mlp"])
        self.s3diff_adapter.vae_fuse_mlp.load_state_dict(sd["state_dict_vae_fuse_mlp"])
        self.s3diff_adapter.unet_fuse_mlp.load_state_dict(sd["state_dict_unet_fuse_mlp"])

        device = self.s3diff_adapter.W.device
        self.s3diff_adapter.W.data.copy_(sd["w"].to(device))

        embeddings = sd["state_embeddings"]
        self.s3diff_adapter.vae_block_embeddings.load_state_dict(embeddings["state_dict_vae_block"])
        self.s3diff_adapter.unet_block_embeddings.load_state_dict(embeddings["state_dict_unet_block"])

        # Update config to reflect actual loaded ranks
        self.s3diff_adapter.config.lora_rank_unet = rank_unet
        self.s3diff_adapter.config.lora_rank_vae = rank_vae
        self._lora_applied = True
        logger.info("S3Diff weights loaded successfully.")

    # ------------------------------------------------------------------
    # Degradation modulation helpers
    # ------------------------------------------------------------------

    def _set_lora_modulation(self, deg_score: torch.Tensor) -> None:
        """Set ``de_mod`` on all LoRA layers based on current degradation scores."""
        rank_vae = self.s3diff_adapter.config.lora_rank_vae
        rank_unet = self.s3diff_adapter.config.lora_rank_unet

        vae_embeds, unet_embeds = self.s3diff_adapter.compute_embeddings(deg_score)

        # Set VAE LoRA modulation
        for layer_name, module in self.vae.named_modules():
            if layer_name not in self._vae_lora_layers:
                continue
            split_name = layer_name.split(".")
            if len(split_name) > 2 and split_name[1] == "down_blocks":
                block_id = int(split_name[2])
                vae_embed = vae_embeds[:, block_id]
            elif len(split_name) > 1 and split_name[1] == "mid_block":
                vae_embed = vae_embeds[:, -2]
            else:
                vae_embed = vae_embeds[:, -1]
            module.de_mod = vae_embed.reshape(-1, rank_vae, rank_vae)

        # Set UNet LoRA modulation
        for layer_name, module in self.unet.named_modules():
            if layer_name not in self._unet_lora_layers:
                continue
            split_name = layer_name.split(".")
            if split_name[0] == "down_blocks":
                block_id = int(split_name[1])
                unet_embed = unet_embeds[:, block_id]
            elif split_name[0] == "mid_block":
                unet_embed = unet_embeds[:, 4]
            elif split_name[0] == "up_blocks":
                block_id = int(split_name[1]) + 5
                unet_embed = unet_embeds[:, block_id]
            else:
                unet_embed = unet_embeds[:, -1]
            module.de_mod = unet_embed.reshape(-1, rank_unet, rank_unet)

    # ------------------------------------------------------------------
    # Prompt encoding
    # ------------------------------------------------------------------

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        num_images_per_prompt: int = 1,
    ) -> torch.Tensor:
        """Encode text prompt(s) into CLIP embeddings."""
        if isinstance(prompt, str):
            prompt = [prompt]

        # Clamp max_length to a reasonable value (default CLIP is 77)
        max_length = min(self.tokenizer.model_max_length, 77)
        tokens = self.tokenizer(
            prompt,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)

        with torch.no_grad():
            embeddings = self.text_encoder(tokens)[0]

        # Repeat for multiple images per prompt
        bs_embed, seq_len, _ = embeddings.shape
        embeddings = embeddings.repeat(1, num_images_per_prompt, 1)
        embeddings = embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)
        return embeddings

    # ------------------------------------------------------------------
    # Main __call__
    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput,
        scale_factor: int = 4,
        pos_prompt: Optional[Union[str, List[str]]] = "",
        neg_prompt: Optional[Union[str, List[str]]] = "",
        guidance_scale: Optional[float] = None,
        degradation_score: Optional[torch.Tensor] = None,
        latent_tiled_size: int = 96,
        latent_tiled_overlap: int = 32,
        num_images_per_prompt: int = 1,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
    ):
        """
        Run one-step image super-resolution with S3Diff.

        Args:
            image ([`~utils.types.ImageInput`]):
                Low-resolution input image(s). Accepted formats: PIL Image, numpy array,
                or torch tensor in ``[0, 1]`` range with shape ``(B, C, H, W)``.
            scale_factor (int):
                Super-resolution scale factor (e.g. 4 for 4× SR). Default: 4.
            pos_prompt (str or List[str]):
                Positive text prompt to guide the reconstruction. Default: ``""``.
            neg_prompt (str or List[str]):
                Negative text prompt for classifier-free guidance. Default: ``""``.
            guidance_scale (float, *optional*):
                CFG scale. If ``None``, defaults to 1.07 (the original S3Diff value).
                Set to ``1.0`` to disable CFG (no negative prompt needed).
            degradation_score (torch.Tensor, *optional*):
                Pre-computed degradation scores of shape ``(B, 2)`` (blur + noise in [0, 1]).
                If ``None`` and ``de_net`` is available, scores are estimated automatically.
                If ``None`` and ``de_net`` is not loaded, all-zero scores are used.
            latent_tiled_size (int):
                Tile size (in latent space) for tiled inference on large images. Default: 96.
            latent_tiled_overlap (int):
                Tile overlap (in latent space). Default: 32.
            num_images_per_prompt (int):
                Number of output images per input image. Default: 1.
            output_type (str):
                Output format: ``"pil"`` or ``"pt"`` (torch tensor). Default: ``"pil"``.
            return_dict (bool):
                If ``True``, return an ``ImagePipelineOutput``. Default: ``True``.
            callback (Callable, *optional*):
                Callback called after the denoising step.
            callback_steps (int):
                Frequency of callback calls. Default: 1.

        Returns:
            [`ImagePipelineOutput`] or ``tuple``:
                A ``ImagePipelineOutput`` with the ``images`` field containing PIL Images,
                or a tuple of ``(images,)`` if ``return_dict`` is ``False``.
        """
        if not self._lora_applied:
            logger.warning(
                "S3Diff LoRA weights have not been loaded. Call `load_s3diff_weights()` first "
                "for best results. Running with untrained (random) weights."
            )

        # Determine effective guidance scale
        effective_guidance_scale = guidance_scale if guidance_scale is not None else 1.07
        use_cfg = effective_guidance_scale > 1.0

        # Determine device and dtype from model parameters
        device = self._execution_device
        dtype = next(self.unet.parameters()).dtype

        # ---- Pre-process input image ---------------------------------
        if isinstance(image, PIL.Image.Image):
            image = [image]

        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            import torchvision.transforms.functional as TF

            image = torch.stack([TF.to_tensor(img.convert("RGB")) for img in image], dim=0)
        elif isinstance(image, np.ndarray):
            if image.ndim == 3:
                image = image[None]
            image = torch.from_numpy(image).permute(0, 3, 1, 2).float() / 255.0

        # At this point `image` should be (B, 3, H, W) in [0, 1]
        image = image.to(device=device, dtype=dtype)
        batch_size = image.shape[0]
        ori_h, ori_w = image.shape[2], image.shape[3]

        # ---- Upscale input to target resolution ----------------------
        image_upscaled = F.interpolate(
            image,
            size=(ori_h * scale_factor, ori_w * scale_factor),
            mode="bilinear",
            align_corners=False,
        ).contiguous()

        # ---- Degradation score estimation ----------------------------
        if degradation_score is not None:
            deg_score = degradation_score.to(device=device, dtype=dtype)
        elif self.de_net is not None:
            with torch.no_grad():
                deg_score = self.de_net(image)
        else:
            deg_score = torch.zeros(batch_size, 2, device=device, dtype=dtype)

        # ---- Set LoRA degradation modulation -------------------------
        if self._lora_applied:
            self._set_lora_modulation(deg_score)

        # ---- Prepare the normalised latent input ----------------------
        image_norm = image_upscaled * 2.0 - 1.0
        image_norm = torch.clamp(image_norm, -1.0, 1.0)
        resize_h, resize_w = image_norm.shape[2], image_norm.shape[3]

        # Pad to multiple of 64 (required by VAE stride)
        pad_h = (math.ceil(resize_h / 64)) * 64 - resize_h
        pad_w = (math.ceil(resize_w / 64)) * 64 - resize_w
        if pad_h > 0 or pad_w > 0:
            # Use 'reflect' when padding < input dimension, otherwise fall back to 'replicate'
            can_reflect = pad_h < resize_h and pad_w < resize_w
            pad_mode = "reflect" if can_reflect else "replicate"
            image_norm = F.pad(image_norm, (0, pad_w, 0, pad_h), mode=pad_mode)

        # ---- Encode prompts ------------------------------------------
        if isinstance(pos_prompt, str):
            pos_prompt = [pos_prompt] * batch_size
        if isinstance(neg_prompt, str):
            neg_prompt = [neg_prompt] * batch_size

        pos_embeds = self._encode_prompt(pos_prompt, device, num_images_per_prompt)
        if use_cfg:
            neg_embeds = self._encode_prompt(neg_prompt, device, num_images_per_prompt)

        # ---- Encode image to latent space ----------------------------
        lq_latent = self.vae.encode(image_norm).latent_dist.sample() * self.vae.config.scaling_factor

        # Set up the scheduler for 1-step inference at timestep 999.
        # We explicitly set the scheduler's timestep list to [999] so that
        # DDPMScheduler.step can look up the previous timestep correctly.
        self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps, device=device)
        timesteps = torch.tensor([999], device=device, dtype=torch.long)
        timesteps = timesteps.expand(batch_size * num_images_per_prompt)

        # ---- UNet prediction (with optional tiling) ------------------
        _, _, h, w = lq_latent.shape
        if h * w <= latent_tiled_size * latent_tiled_size:
            pos_model_pred = self.unet(lq_latent, timesteps, encoder_hidden_states=pos_embeds).sample
            if use_cfg:
                neg_model_pred = self.unet(lq_latent, timesteps, encoder_hidden_states=neg_embeds).sample
        else:
            pos_model_pred, neg_model_pred = self._tiled_unet_forward(
                lq_latent,
                timesteps,
                pos_embeds,
                neg_embeds if use_cfg else None,
                latent_tiled_size,
                latent_tiled_overlap,
            )

        # Apply CFG
        if use_cfg:
            model_pred = neg_model_pred + effective_guidance_scale * (pos_model_pred - neg_model_pred)
        else:
            model_pred = pos_model_pred

        # Callback
        if callback is not None and 0 % callback_steps == 0:
            callback(0, 1, lq_latent)

        # ---- Denoising step ------------------------------------------
        x_denoised = self.scheduler.step(model_pred, timesteps[0], lq_latent, return_dict=True).prev_sample

        # ---- Decode to pixel space -----------------------------------
        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1.0, 1.0)

        # Crop back to the target size (remove padding)
        output_image = output_image[:, :, :resize_h, :resize_w]

        # Convert to [0, 1]
        output_image = output_image * 0.5 + 0.5

        if output_type == "pil":
            import torchvision.transforms.functional as TF

            output = [TF.to_pil_image(img.clamp(0, 1)) for img in output_image]
        else:
            output = output_image

        if not return_dict:
            return (output,)

        return ImagePipelineOutput(images=output)

    # ------------------------------------------------------------------
    # Tiled UNet forward (for large images)
    # ------------------------------------------------------------------

    def _gaussian_weights(
        self, tile_width: int, tile_height: int, nbatches: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Generate a 2-D Gaussian weight mask for tile blending."""
        var = 0.01
        midpoint_x = (tile_width - 1) / 2.0
        midpoint_y = tile_height / 2.0
        x_probs = [
            math.exp(-((x - midpoint_x) ** 2) / (tile_width**2) / (2 * var))
            / math.sqrt(2 * math.pi * var)
            for x in range(tile_width)
        ]
        y_probs = [
            math.exp(-((y - midpoint_y) ** 2) / (tile_height**2) / (2 * var))
            / math.sqrt(2 * math.pi * var)
            for y in range(tile_height)
        ]
        weights = torch.tensor(np.outer(y_probs, x_probs), dtype=dtype, device=device)
        return weights.unsqueeze(0).unsqueeze(0).expand(nbatches, self.unet.config.in_channels, -1, -1)

    def _tiled_unet_forward(
        self,
        lq_latent: torch.Tensor,
        timesteps: torch.Tensor,
        pos_embeds: torch.Tensor,
        neg_embeds: Optional[torch.Tensor],
        tile_size: int,
        tile_overlap: int,
    ):
        """Run UNet inference using tiling for large latents.

        Returns:
            Tuple of ``(pos_model_pred, neg_model_pred)`` where ``neg_model_pred``
            is ``None`` when ``neg_embeds`` is ``None``.
        """
        _, _, h, w = lq_latent.shape
        tile_size = min(tile_size, min(h, w))
        tile_weights = self._gaussian_weights(tile_size, tile_size, 1, lq_latent.device, lq_latent.dtype)

        def _compute_grid(dim_size):
            count, cur = 0, 0
            while cur < dim_size:
                cur = max(count * tile_size - tile_overlap * count, 0) + tile_size
                count += 1
            return count

        grid_cols = _compute_grid(h)
        grid_rows = _compute_grid(w)

        pos_noise_pred = torch.zeros_like(lq_latent)
        neg_noise_pred = torch.zeros_like(lq_latent) if neg_embeds is not None else None
        contributors = torch.zeros_like(lq_latent)

        for row in range(grid_rows):
            for col in range(grid_cols):
                ofs_x = max(row * tile_size - tile_overlap * row, 0)
                ofs_y = max(col * tile_size - tile_overlap * col, 0)
                if row == grid_rows - 1:
                    ofs_x = w - tile_size
                if col == grid_cols - 1:
                    ofs_y = h - tile_size

                sx, ex = ofs_x, ofs_x + tile_size
                sy, ey = ofs_y, ofs_y + tile_size
                tile = lq_latent[:, :, sy:ey, sx:ex]

                pos_out = self.unet(tile, timesteps, encoder_hidden_states=pos_embeds).sample
                pos_noise_pred[:, :, sy:ey, sx:ex] += pos_out * tile_weights

                if neg_embeds is not None:
                    neg_out = self.unet(tile, timesteps, encoder_hidden_states=neg_embeds).sample
                    neg_noise_pred[:, :, sy:ey, sx:ex] += neg_out * tile_weights

                contributors[:, :, sy:ey, sx:ex] += tile_weights

        pos_noise_pred /= contributors
        if neg_noise_pred is not None:
            neg_noise_pred /= contributors

        return pos_noise_pred, neg_noise_pred
