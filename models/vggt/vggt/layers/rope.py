# coding=utf-8
# Adapted from
# https://github.com/facebookresearch/vggt/
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.


# Implementation of 2D Rotary Position Embeddings (RoPE).

# This module provides a clean implementation of 2D Rotary Position Embeddings,
# which extends the original RoPE concept to handle 2D spatial positions.

# Inspired by:
#         https://github.com/meta-llama/codellama/blob/main/llama/model.py
#         https://github.com/naver-ai/rope-vit


from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu


class PositionGetter:
    """Generates and caches 2D spatial positions for patches in a grid.

    This class efficiently manages the generation of spatial coordinates for patches
    in a 2D grid, caching results to avoid redundant computations.

    Attributes:
        position_cache: Dictionary storing precomputed position tensors for different
            grid dimensions.
    """

    def __init__(self):
        """Initializes the position generator with an empty cache."""
        self.position_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def __call__(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Generates spatial positions for a batch of patches.

        Args:
            batch_size: Number of samples in the batch.
            height: Height of the grid in patches.
            width: Width of the grid in patches.
            device: Target device for the position tensor.

        Returns:
            Tensor of shape (batch_size, height*width, 2) containing y,x coordinates
            for each position in the grid, repeated for each batch item.
        """
        if (height, width) not in self.position_cache:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            positions = torch.cartesian_prod(y_coords, x_coords)
            self.position_cache[height, width] = positions

        cached_positions = self.position_cache[height, width]
        return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()


class RotaryPositionEmbedding2D(nn.Module):
    """2D Rotary Position Embedding implementation.

    This module applies rotary position embeddings to input tokens based on their
    2D spatial positions. It handles the position-dependent rotation of features
    separately for vertical and horizontal dimensions.

    Args:
        frequency: Base frequency for the position embeddings. Default: 100.0
        scaling_factor: Scaling factor for frequency computation. Default: 1.0

    Attributes:
        base_frequency: Base frequency for computing position embeddings.
        scaling_factor: Factor to scale the computed frequencies.
        frequency_cache: Cache for storing precomputed frequency components.
    """

    def __init__(self, frequency: float = 100.0, scaling_factor: float = 1.0):
        """Initializes the 2D RoPE module."""
        super().__init__()
        self.base_frequency = frequency
        self.scaling_factor = scaling_factor
        self.frequency_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.cos_sin_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.max_position_cache: Dict[Tuple, int] = {}

    def _compute_frequency_components(
        self, token_attribute: List, input_positions: torch.tensor, height: None, width: None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes frequency components for rotary embeddings.

        Args:
            token_attribute: [dim: Feature dimension (must be even), 
                device: Target device for computations, 
                dtype: Data type for the computed tensors.]
            input_positions: Position indices.
            height: height of the image
            width: height of the image
        Returns:
            Tuple of (cosine, sine) tensors for frequency components.
        """
        dim, device, dtype = token_attribute[0], token_attribute[1], token_attribute[2]
        batch_size = input_positions.shape[0]
        if height is None or width is None:
            seq_len = int(input_positions.max()) + 1
        else:
            hw_key = (height, width)
            if hw_key not in self.max_position_cache:
                self.max_position_cache[hw_key] = int(input_positions.max()) + 1
            seq_len = self.max_position_cache[hw_key]
        cache_key = (dim, seq_len, device, dtype)
        if cache_key not in self.frequency_cache:
            # Compute frequency bands
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency**exponents)
            # Generate position-dependent frequencies
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq)
            # Compute and cache frequency components
            angles = angles.to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            cos_components = angles.cos().to(dtype)
            sin_components = angles.sin().to(dtype)
            self.frequency_cache[cache_key] = (cos_components, sin_components)
        cos_components, sin_components = self.frequency_cache[cache_key]
        if height is None or width is None or batch_size is None:
            vertical_cos = F.embedding(input_positions[..., 0], cos_components)[:, None, :, :]
            vertical_sin = F.embedding(input_positions[..., 0], sin_components)[:, None, :, :]
            horizontal_cos = F.embedding(input_positions[..., 1], cos_components)[:, None, :, :]
            horizontal_sin = F.embedding(input_positions[..., 1], sin_components)[:, None, :, :]
            cos_sin_output = [vertical_cos, vertical_sin, horizontal_cos, horizontal_sin]
            return cos_sin_output
        sub_cache_key = (height, width, batch_size)
        if sub_cache_key not in self.cos_sin_cache:
            # Embed positions with frequency components
            vertical_cos = F.embedding(input_positions[..., 0], cos_components)[:, None, :, :]
            vertical_sin = F.embedding(input_positions[..., 0], sin_components)[:, None, :, :]
            horizontal_cos = F.embedding(input_positions[..., 1], cos_components)[:, None, :, :]
            horizontal_sin = F.embedding(input_positions[..., 1], sin_components)[:, None, :, :]
            self.cos_sin_cache[sub_cache_key] = (vertical_cos, vertical_sin, horizontal_cos, horizontal_sin)
        vertical_cos, vertical_sin, horizontal_cos, horizontal_sin = self.cos_sin_cache[sub_cache_key]
        cos_sin_output = [vertical_cos, vertical_sin, horizontal_cos, horizontal_sin]
        return cos_sin_output

    @staticmethod
    def _rotate_features(x: torch.Tensor) -> torch.Tensor:
        """Performs feature rotation by splitting and recombining feature dimensions.

        Args:
            x: Input tensor to rotate.

        Returns:
            Rotated feature tensor.
        """
        feature_dim = x.shape[-1]
        x1, x2 = x[..., : feature_dim // 2], x[..., feature_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d_rope(
        self, tokens: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Applies 1D rotary position embeddings along one dimension.

        Args:
            tokens: Input token features.
            cos: Cosine components for rotation.
            sin: Sine components for rotation.

        Returns:
            Tokens with applied rotary position embeddings.
        """
        # Apply rotation
        return torch_npu.npu_rotary_mul(tokens, cos, sin, rotary_mode='half')

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Applies 2D rotary position embeddings to input tokens.

        Args:
            tokens: Input tensor of shape (batch_size, n_heads, n_tokens, dim).
                   The feature dimension (dim) must be divisible by 4.
            positions: Position tensor of shape (batch_size, n_tokens, 2) containing
                      the y and x coordinates for each token.
            height: height of the image
            width: height of the image

        Returns:
            Tensor of same shape as input with applied 2D rotary position embeddings.

        Raises:
            AssertionError: If input dimensions are invalid or positions are malformed.
        """
        # Validate inputs
        assert tokens.size(-1) % 2 == 0, "Feature dimension must be even"
        assert positions.ndim == 3 and positions.shape[-1] == 2, "Positions must have shape (batch_size, n_tokens, 2)"
        # Compute feature dimension for each spatial direction
        feature_dim = tokens.size(-1) // 2
        token_attribute = [feature_dim, tokens.device, tokens.dtype]
        # Get frequency components
        cos_sin_output = self._compute_frequency_components(token_attribute, positions, height, width)
        vertical_cos, vertical_sin, horizontal_cos, horizontal_sin \
            = cos_sin_output[0], cos_sin_output[1], cos_sin_output[2], cos_sin_output[3]
        # Split features for vertical and horizontal processing
        vertical_features, horizontal_features = tokens.chunk(2, dim=-1)
        # Apply RoPE separately for each dimension
        vertical_features = self._apply_1d_rope(vertical_features, vertical_cos, vertical_sin)
        horizontal_features = self._apply_1d_rope(horizontal_features, horizontal_cos, horizontal_sin)
        # Combine processed features
        return torch.cat((vertical_features, horizontal_features), dim=-1)
