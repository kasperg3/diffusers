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

"""Degradation Estimation Network (DEResNet) for S3Diff image super-resolution."""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...models.modeling_utils import ModelMixin


def default_init_weights(module_list, scale=1, bias_fill=0):
    """Initialize network weights with orthogonal method.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual blocks.
            Default: 1.
        bias_fill (float): The value to fill bias. Default: 0.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


class ResidualBlockNoBN(nn.Module):
    """Residual block without batch normalization.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features. Default: 64.
        res_scale (float): Residual scale. Default: 1.
    """

    def __init__(self, num_feat=64, res_scale=1):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        # initialization
        default_init_weights([self.conv1, self.conv2], scale=0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class DEResNet(nn.Module):
    """Degradation Estimator with ResNet (no BN) architecture.

    Estimates degradation scores (e.g., blur and noise levels) from an input image.
    Based on the paper 'Degradation-Guided One-Step Image Super-Resolution with Diffusion Priors'.

    Args:
        num_in_ch (int): Number of input channels. Default: 3.
        num_degradation (int): Number of degradation types to estimate. Default: 2 (blur + noise).
        degradation_degree_actv (str): Activation function for degradation degree output.
            Options: 'sigmoid' or 'tanh'. Default: 'sigmoid'.
        num_feats (list[int]): Channel numbers for each stage. Default: [64, 64, 64, 128].
        num_blocks (list[int]): Number of residual blocks per stage. Default: [2, 2, 2, 2].
        downscales (list[int]): Downscale factors per stage (1=no downscale, 2=2x). Default: [1, 1, 2, 1].
    """

    def __init__(
        self,
        num_in_ch=3,
        num_degradation=2,
        degradation_degree_actv="sigmoid",
        num_feats=None,
        num_blocks=None,
        downscales=None,
    ):
        super().__init__()

        if num_feats is None:
            num_feats = [64, 64, 64, 128]
        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]
        if downscales is None:
            downscales = [1, 1, 2, 1]

        assert len(num_feats) == len(num_blocks) == len(downscales)

        num_stage = len(num_feats)
        self.num_degradation = num_degradation

        # First convolution layer for each degradation branch
        self.conv_first = nn.ModuleList()
        for _ in range(num_degradation):
            self.conv_first.append(nn.Conv2d(num_in_ch, num_feats[0], 3, 1, 1))

        # Body (residual blocks + downsampling) for each degradation branch
        self.body = nn.ModuleList()
        for _ in range(num_degradation):
            body = []
            for stage in range(num_stage):
                for _ in range(num_blocks[stage]):
                    body.append(ResidualBlockNoBN(num_feats[stage]))
                if downscales[stage] == 1:
                    if stage < num_stage - 1 and num_feats[stage] != num_feats[stage + 1]:
                        body.append(nn.Conv2d(num_feats[stage], num_feats[stage + 1], 3, 1, 1))
                elif downscales[stage] == 2:
                    body.append(nn.Conv2d(num_feats[stage], num_feats[min(stage + 1, num_stage - 1)], 3, 2, 1))
                else:
                    raise ValueError(f"Unsupported downscale factor {downscales[stage]}, only 1 and 2 are supported.")
            self.body.append(nn.Sequential(*body))

        # Fully-connected layers for degree prediction
        if degradation_degree_actv == "sigmoid":
            actv = nn.Sigmoid
        elif degradation_degree_actv == "tanh":
            actv = nn.Tanh
        else:
            raise ValueError(
                f"Only 'sigmoid' and 'tanh' are supported for degradation_degree_actv, "
                f"got '{degradation_degree_actv}'."
            )

        self.fc_degree = nn.ModuleList()
        for _ in range(num_degradation):
            self.fc_degree.append(
                nn.Sequential(
                    nn.Linear(num_feats[-1], 512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, 1),
                    actv(),
                )
            )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Initialize weights
        default_init_weights([self.conv_first, self.body, self.fc_degree], scale=0.1)

    def _clone_module(self, module):
        return copy.deepcopy(module)

    def _average_parameters(self, modules):
        avg_module = self._clone_module(modules[0])
        for name, param in avg_module.named_parameters():
            avg_param = sum(mod.state_dict()[name].data for mod in modules) / len(modules)
            param.data.copy_(avg_param)
        return avg_module

    def expand_degradation_modules(self, new_num_degradation):
        """Expand the model to handle more degradation types."""
        if new_num_degradation <= self.num_degradation:
            return
        for modules in [self.conv_first, self.body, self.fc_degree]:
            avg_module = self._average_parameters(list(modules)[:2])
            while len(modules) < new_num_degradation:
                modules.append(self._clone_module(avg_module))
        self.num_degradation = new_num_degradation

    def load_model(self, path):
        """Load model weights from a checkpoint file."""
        state_dict = torch.load(path, map_location="cpu")
        self.load_state_dict(state_dict, strict=True)

    def forward(self, x):
        """
        Forward pass to estimate degradation scores.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W), values in [0, 1].

        Returns:
            torch.Tensor: Degradation scores of shape (B, num_degradation).
        """
        degrees = []
        for i in range(self.num_degradation):
            x_out = self.conv_first[i](x)
            feat = self.body[i](x_out)
            feat = self.avg_pool(feat)
            feat = feat.squeeze(-1).squeeze(-1)
            degrees.append(self.fc_degree[i](feat).squeeze(-1))
        return torch.stack(degrees, dim=1)
