<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# S3Diff

S3Diff was proposed in [S3Diff: Degradation-Guided One-Step Image Super-Resolution with Diffusion Priors](https://huggingface.co/papers/2409.17058) by Aiping Zhang, Zhen Xing, Mingjia Li, Zuxuan Wu, and Yu-Gang Jiang.

The abstract from the paper is:

*Image super-resolution (SR) is a fundamental task in computer vision. Diffusion-based SR methods offer a promising paradigm for generating visually appealing results but are limited by slow inference and the requirement of large computational resources. Recently, single-step SR methods have been proposed to address this challenge. However, they still lack flexibility in handling varying degradation levels. In this paper, we propose S3Diff, a degradation-guided one-step image super-resolution method. By incorporating degree-specific information to guide the diffusion process, S3Diff is capable of adaptively handling different degradation levels and generating photo-realistic super-resolution images in a single denoising step. Specifically, we first pre-train a Degradation Estimation Network (DEResNet) to predict the degradation degrees of low-resolution images. The estimated degradation information is then injected into the base model through novel degradation-guided LoRA modules to steer the denoising process. Extensive experiments demonstrate that S3Diff achieves state-of-the-art performance on both synthetic and real-world benchmarks, especially at the hardest degradation levels, while maintaining competitive efficiency.*

The original codebase can be found at [ArcticHare105/S3Diff](https://github.com/ArcticHare105/S3Diff), and additional checkpoints are available at [zhangap/S3Diff](https://huggingface.co/zhangap/S3Diff).

## Tips

- S3Diff builds on top of [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo) and adds degradation-guided LoRA adapters plus a degradation estimation network.
- The pipeline performs super-resolution in a **single denoising step**, making it very fast.
- Use `pipe.load_s3diff_weights("zhangap/S3Diff")` to load the LoRA adapter weights (requires [PEFT](https://github.com/huggingface/peft)).
- Use `pipe.load_de_net_weights("zhangap/S3Diff")` to load the degradation estimation network, which automatically computes degradation scores from the input image.
- When `de_net` is not loaded, you can pass pre-computed degradation scores via the `degradation_score` argument, or rely on all-zero scores (which still produces reasonable results).

```python
import torch
from PIL import Image
from diffusers import S3DiffPipeline

# Load base SD-Turbo model
pipe = S3DiffPipeline.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

# Load S3Diff adapter weights and degradation estimation network
pipe.load_s3diff_weights("zhangap/S3Diff")
pipe.load_de_net_weights("zhangap/S3Diff")

# Run 4× super-resolution
lr_image = Image.open("low_res.png").convert("RGB")
result = pipe(
    image=lr_image,
    scale_factor=4,
    pos_prompt="high quality, detailed",
    neg_prompt="blurry, low quality",
).images[0]
result.save("high_res.png")
```

## S3DiffPipeline

[[autodoc]] S3DiffPipeline
    - all
    - __call__
    - load_s3diff_weights
    - load_de_net_weights

## S3DiffAdapter

[[autodoc]] S3DiffAdapter
    - all

## DEResNet

[[autodoc]] DEResNet
    - all
    - forward

## ImagePipelineOutput

[[autodoc]] pipelines.ImagePipelineOutput
