# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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

import json
import os
import tempfile
import unittest

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DDPMScheduler, S3DiffPipeline, UNet2DConditionModel
from diffusers.pipelines.s3diff import DEResNet, S3DiffAdapter
from diffusers.utils.testing_utils import enable_full_determinism, require_torch, torch_device


enable_full_determinism()


def _create_dummy_tokenizer():
    """Create a minimal CLIPTokenizer from scratch without network access."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab = {"<|startoftext|>": 0, "<|endoftext|>": 1, "!": 2, "a</w>": 3, "b</w>": 4}
        vocab_path = os.path.join(tmpdir, "vocab.json")
        with open(vocab_path, "w") as f:
            json.dump(vocab, f)
        merges_path = os.path.join(tmpdir, "merges.txt")
        with open(merges_path, "w") as f:
            f.write("#version: 0.2\n")
        tokenizer = CLIPTokenizer(vocab_file=vocab_path, merges_file=merges_path, model_max_length=77)
    return tokenizer


def get_dummy_components():
    torch.manual_seed(0)
    unet = UNet2DConditionModel(
        block_out_channels=(32, 64),
        layers_per_block=2,
        sample_size=32,
        in_channels=4,
        out_channels=4,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=32,
    )
    torch.manual_seed(0)
    vae = AutoencoderKL(
        block_out_channels=[32, 64],
        in_channels=3,
        out_channels=3,
        down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
        up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
        latent_channels=4,
    )
    torch.manual_seed(0)
    scheduler = DDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        prediction_type="epsilon",
    )
    text_encoder_config = CLIPTextConfig(
        bos_token_id=0,
        eos_token_id=2,
        hidden_size=32,
        intermediate_size=37,
        layer_norm_eps=1e-05,
        num_attention_heads=4,
        num_hidden_layers=5,
        pad_token_id=1,
        vocab_size=1000,
    )
    text_encoder = CLIPTextModel(text_encoder_config)
    tokenizer = _create_dummy_tokenizer()

    components = {
        "unet": unet,
        "vae": vae,
        "scheduler": scheduler,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "s3diff_adapter": S3DiffAdapter(lora_rank_unet=8, lora_rank_vae=4),
        "de_net": None,
    }
    return components


def get_dummy_inputs(device, seed=0):
    generator = torch.Generator(device=device).manual_seed(seed)
    image = torch.randn((1, 3, 32, 32), generator=generator, device=device).clamp(0, 1)
    return {
        "image": image,
        "scale_factor": 1,
        "pos_prompt": "high quality, detailed",
        "neg_prompt": "blurry, low quality",
        "guidance_scale": 1.0,
        "output_type": "pt",
    }


@require_torch
class S3DiffPipelineFastTests(unittest.TestCase):
    """Fast unit tests for S3DiffPipeline that run without network access."""

    def test_pipeline_can_be_instantiated(self):
        """Test that the pipeline can be created with dummy components."""
        components = get_dummy_components()
        pipe = S3DiffPipeline(**components)
        self.assertIsInstance(pipe, S3DiffPipeline)

    def test_pipeline_has_expected_components(self):
        """Test that the pipeline registers all expected module components."""
        components = get_dummy_components()
        pipe = S3DiffPipeline(**components)
        self.assertTrue(hasattr(pipe, "vae"))
        self.assertTrue(hasattr(pipe, "unet"))
        self.assertTrue(hasattr(pipe, "text_encoder"))
        self.assertTrue(hasattr(pipe, "tokenizer"))
        self.assertTrue(hasattr(pipe, "scheduler"))
        self.assertTrue(hasattr(pipe, "de_net"))
        self.assertTrue(hasattr(pipe, "s3diff_adapter"))

    def test_pipeline_forward_no_lora(self):
        """Test forward pass runs without LoRA weights (uses random S3Diff params)."""
        components = get_dummy_components()
        pipe = S3DiffPipeline(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=True)

        inputs = get_dummy_inputs(torch_device)
        output = pipe(**inputs)

        self.assertIsInstance(output, type(pipe(return_dict=True, **get_dummy_inputs(torch_device))))
        images = output.images
        self.assertEqual(images.shape[0], 1)  # batch size
        self.assertEqual(images.shape[1], 3)  # channels

    def test_pipeline_output_shape_with_scale_factor(self):
        """Test that the output has the correct spatial dimensions."""
        components = get_dummy_components()
        pipe = S3DiffPipeline(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=True)

        inputs = get_dummy_inputs(torch_device)
        inputs["scale_factor"] = 2
        output = pipe(**inputs)

        images = output.images
        # Input is 32x32, scale_factor=2 → output should be 64x64
        self.assertEqual(images.shape[-2], 64)
        self.assertEqual(images.shape[-1], 64)

    def test_pipeline_output_type_pil(self):
        """Test PIL output type."""
        import PIL.Image

        components = get_dummy_components()
        pipe = S3DiffPipeline(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=True)

        inputs = get_dummy_inputs(torch_device)
        inputs["output_type"] = "pil"
        output = pipe(**inputs)

        self.assertIsInstance(output.images[0], PIL.Image.Image)

    def test_pipeline_pil_image_input(self):
        """Test pipeline with a PIL Image as input."""
        import PIL.Image

        components = get_dummy_components()
        pipe = S3DiffPipeline(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=True)

        pil_image = PIL.Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        output = pipe(
            image=pil_image,
            scale_factor=1,
            output_type="pt",
        )

        self.assertEqual(output.images.shape[0], 1)

    def test_pipeline_no_cfg(self):
        """Test pipeline with guidance_scale=1.0 (no CFG)."""
        components = get_dummy_components()
        pipe = S3DiffPipeline(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=True)

        inputs = get_dummy_inputs(torch_device)
        inputs["guidance_scale"] = 1.0
        output = pipe(**inputs)

        self.assertIsNotNone(output.images)

    def test_pipeline_cfg(self):
        """Test pipeline with guidance_scale > 1.0 (with CFG)."""
        components = get_dummy_components()
        pipe = S3DiffPipeline(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=True)

        inputs = get_dummy_inputs(torch_device)
        inputs["guidance_scale"] = 1.07
        output = pipe(**inputs)

        self.assertIsNotNone(output.images)

    def test_pipeline_return_tuple(self):
        """Test that return_dict=False returns a tuple."""
        components = get_dummy_components()
        pipe = S3DiffPipeline(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=True)

        inputs = get_dummy_inputs(torch_device)
        inputs["return_dict"] = False
        output = pipe(**inputs)

        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output), 1)

    def test_pipeline_with_degradation_score(self):
        """Test pipeline when degradation scores are passed directly."""
        components = get_dummy_components()
        pipe = S3DiffPipeline(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=True)

        inputs = get_dummy_inputs(torch_device)
        inputs["degradation_score"] = torch.tensor([[0.5, 0.3]], device=torch_device)
        output = pipe(**inputs)

        self.assertIsNotNone(output.images)

    def test_pipeline_with_de_net(self):
        """Test pipeline with the DEResNet degradation estimator attached."""
        components = get_dummy_components()
        components["de_net"] = DEResNet(num_in_ch=3, num_degradation=2)
        pipe = S3DiffPipeline(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=True)

        inputs = get_dummy_inputs(torch_device)
        output = pipe(**inputs)

        self.assertIsNotNone(output.images)


@require_torch
class DEResNetTests(unittest.TestCase):
    """Unit tests for the DEResNet degradation estimation model."""

    def test_de_resnet_instantiation(self):
        """Test that DEResNet can be instantiated with default parameters."""
        model = DEResNet(num_in_ch=3, num_degradation=2)
        self.assertIsInstance(model, DEResNet)

    def test_de_resnet_forward(self):
        """Test DEResNet forward pass produces correct output shape."""
        model = DEResNet(num_in_ch=3, num_degradation=2).to(torch_device)
        model.eval()

        x = torch.randn(2, 3, 64, 64).to(torch_device)
        with torch.no_grad():
            output = model(x)

        # Output should be (B, num_degradation)
        self.assertEqual(output.shape, (2, 2))
        # Values should be in [0, 1] since sigmoid activation is used
        self.assertTrue((output >= 0).all())
        self.assertTrue((output <= 1).all())

    def test_de_resnet_output_range(self):
        """Test that DEResNet outputs are in [0, 1] with default sigmoid activation."""
        model = DEResNet(num_in_ch=3, num_degradation=2).eval()

        x = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            output = model(x)

        self.assertTrue(output.min() >= 0)
        self.assertTrue(output.max() <= 1)

    def test_de_resnet_single_degradation(self):
        """Test DEResNet with a single degradation type."""
        model = DEResNet(num_in_ch=3, num_degradation=1).eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            output = model(x)
        self.assertEqual(output.shape, (1, 1))

    def test_de_resnet_batch_consistency(self):
        """Test that processing a batch gives the same result as processing individually."""
        torch.manual_seed(42)
        model = DEResNet(num_in_ch=3, num_degradation=2).eval()

        x1 = torch.randn(1, 3, 64, 64)
        x2 = torch.randn(1, 3, 64, 64)
        x_batch = torch.cat([x1, x2], dim=0)

        with torch.no_grad():
            out1 = model(x1)
            out2 = model(x2)
            out_batch = model(x_batch)

        self.assertTrue(torch.allclose(out1, out_batch[:1], atol=1e-5))
        self.assertTrue(torch.allclose(out2, out_batch[1:], atol=1e-5))
