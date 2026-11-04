"""ClinStab: Medformer-style multi-granularity classification with SCDM.

This module is a self-contained implementation of the equations in the ClinStab
paper.  It keeps the sensor/lead axis explicit after patch embedding, routes
the highest-energy channels through attention, routes the remaining channels
through a depth-wise temporal convolution, and scatters both streams back to
their original channel positions before residual fusion.
"""

from __future__ import annotations

from math import ceil, isfinite
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _compatible_heads(d_model: int, requested: int) -> int:
    """Return the largest requested head count that divides ``d_model``."""

    requested = max(1, min(int(requested), int(d_model)))
    for heads in range(requested, 0, -1):
        if d_model % heads == 0:
            return heads
    return 1


class _PatchEmbedding(nn.Module):
    """Channel-aligned non-overlapping patch embedding for one granularity."""

    def __init__(
        self,
        seq_len: int,
        patch_len: int,
        d_model: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if patch_len < 1:
            raise ValueError("patch_len must be positive")
        self.patch_len = int(patch_len)
        max_patches = max(1, ceil(max(1, int(seq_len)) / self.patch_len))
        self.value_projection = nn.Linear(self.patch_len, d_model)
        self.position = nn.Parameter(torch.zeros(1, max_patches, 1, d_model))
        self.granularity = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        self.dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.granularity, std=0.02)

    def _position_for(self, num_patches: int) -> torch.Tensor:
        if num_patches <= self.position.size(1):
            return self.position[:, :num_patches]
        # The configured sequence length normally covers all inputs.  Linear
        # interpolation keeps the module usable for a longer evaluation window.
        pos = self.position.squeeze(0).squeeze(1).transpose(0, 1).unsqueeze(0)
        pos = F.interpolate(pos, size=num_patches, mode="linear", align_corners=False)
        return pos.transpose(1, 2).unsqueeze(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, channels] -> [batch, patches, channels, hidden]
        batch, time, channels = x.shape
        pad = (-time) % self.patch_len
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        num_patches = x.size(1) // self.patch_len
        patches = x.view(batch, num_patches, self.patch_len, channels)
        patches = patches.permute(0, 1, 3, 2).contiguous()
        embedded = self.value_projection(patches)
        embedded = embedded + self._position_for(num_patches) + self.granularity
        return self.dropout(embedded)


class _SalienceAttention(nn.Module):
    """Self-attention over temporal patches for each selected channel."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        heads = _compatible_heads(d_model, n_heads)
        self.attention = nn.MultiheadAttention(
            d_model, heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, patches, selected_channels, hidden]
        batch, patches, channels, hidden = x.shape
        if channels == 0:
            return x
        sequence = x.permute(0, 2, 1, 3).reshape(batch * channels, patches, hidden)
        attended, _ = self.attention(sequence, sequence, sequence, need_weights=False)
        sequence = self.norm1(sequence + self.dropout(attended))
        sequence = self.norm2(sequence + self.dropout(self.feed_forward(sequence)))
        return sequence.view(batch, channels, patches, hidden).permute(0, 2, 1, 3)


class _SCDMBlock(nn.Module):
    """One channel-preserving Salience-Context Dual Modulation block."""

    def __init__(
        self,
        channels: int,
        d_model: int,
        n_heads: int,
        dropout: float,
        alpha: float,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("enc_in/channels must be positive")
        if alpha <= 0:
            raise ValueError("scdm_alpha must be positive")
        self.channels = int(channels)
        self.d_model = int(d_model)
        self.k = max(1, min(self.channels, int(self.channels / alpha)))
        self.context_channels = self.channels - self.k
        self.salience = _SalienceAttention(d_model, n_heads, dropout)
        if self.context_channels:
            context_width = self.context_channels * d_model
            self.context_conv = nn.Conv1d(
                context_width,
                context_width,
                kernel_size=3,
                padding=1,
                groups=context_width,
                bias=False,
            )
            self.context_norm = nn.LayerNorm(d_model)
        else:
            self.context_conv = None
            self.context_norm = None
        self.output_projection = nn.Linear(d_model, d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _gather_channels(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        batch, patches, _, hidden = x.shape
        expanded = indices[:, None, :, None].expand(batch, patches, -1, hidden)
        return torch.gather(x, dim=2, index=expanded)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, patches, channels, hidden]
        batch, patches, channels, hidden = x.shape
        if channels != self.channels or hidden != self.d_model:
            raise ValueError(
                "SCDM received shape with channels/hidden={} / {}; expected {} / {}".format(
                    channels, hidden, self.channels, self.d_model
                )
            )

        # The ranking is deliberately detached: gradients flow through both
        # selected value streams, but not through the discrete TopK indices.
        energy = x.detach().pow(2).sum(dim=-1).mean(dim=1)
        salience_indices = torch.topk(
            energy, k=self.k, dim=-1, largest=True, sorted=False
        ).indices
        salience_indices = salience_indices.detach()
        salience = self._gather_channels(x, salience_indices)
        salience = self.salience(salience)

        if self.context_channels:
            selected = torch.zeros(
                batch, channels, dtype=torch.bool, device=x.device
            )
            selected.scatter_(1, salience_indices, True)
            all_indices = torch.arange(channels, device=x.device).expand(batch, -1)
            context_indices = all_indices.masked_select(~selected).view(
                batch, self.context_channels
            )
            context = self._gather_channels(x, context_indices)
            context_sequence = context.permute(0, 2, 3, 1).reshape(
                batch, self.context_channels * hidden, patches
            )
            context_sequence = self.context_conv(context_sequence)
            context = context_sequence.view(
                batch, self.context_channels, hidden, patches
            ).permute(0, 3, 1, 2)
            context = self.context_norm(context + self._gather_channels(x, context_indices))
        else:
            context_indices = torch.empty(
                batch, 0, dtype=torch.long, device=x.device
            )
            context = x[:, :, :0, :]

        routed = torch.zeros_like(x)
        salience_scatter = salience_indices[:, None, :, None].expand(
            batch, patches, self.k, hidden
        )
        routed.scatter_(2, salience_scatter, salience)
        if self.context_channels:
            context_scatter = context_indices[:, None, :, None].expand(
                batch, patches, self.context_channels, hidden
            )
            routed.scatter_(2, context_scatter, context)

        output = self.output_projection(routed)
        output = self.output_norm(x + self.dropout(output))
        return output, salience_indices


class Model(nn.Module):
    """ClinStab classifier with the same interface as the Medformer models."""

    def __init__(self, configs) -> None:
        super().__init__()
        self.task_name = getattr(configs, "task_name", "classification")
        if self.task_name != "classification":
            raise ValueError("ClinStab currently supports classification only")
        self.seq_len = int(getattr(configs, "seq_len", 256))
        self.enc_in = int(getattr(configs, "enc_in", 1))
        self.d_model = int(getattr(configs, "d_model", 128))
        self.n_heads = int(getattr(configs, "n_heads", 8))
        self.dropout = float(getattr(configs, "dropout", 0.1))
        self.num_class = int(getattr(configs, "num_class", 2))
        if self.seq_len < 1:
            raise ValueError("seq_len must be positive")
        if self.enc_in < 1:
            raise ValueError("enc_in must be positive")
        if self.d_model < 1:
            raise ValueError("d_model must be positive")
        if self.n_heads < 1:
            raise ValueError("n_heads must be positive")
        if self.num_class < 2:
            raise ValueError("num_class must be at least 2")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        patch_spec = getattr(configs, "patch_len_list", "2,4,8")
        self.patch_len_list = [
            int(item.strip())
            for item in str(patch_spec).split(",")
            if item.strip()
        ]
        if not self.patch_len_list:
            raise ValueError("patch_len_list must contain at least one patch length")
        alpha = float(getattr(configs, "scdm_alpha", 4.67))
        if not isfinite(alpha) or alpha <= 0:
            raise ValueError("scdm_alpha must be a finite positive number")
        layers = max(1, int(getattr(configs, "e_layers", 1)))

        self.patch_embeddings = nn.ModuleList(
            [
                _PatchEmbedding(self.seq_len, patch_len, self.d_model, self.dropout)
                for patch_len in self.patch_len_list
            ]
        )
        self.scdm_layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        _SCDMBlock(
                            self.enc_in,
                            self.d_model,
                            self.n_heads,
                            self.dropout,
                            alpha,
                        )
                        for _ in range(layers)
                    ]
                )
                for _ in self.patch_len_list
            ]
        )
        representation_width = self.d_model * len(self.patch_len_list)
        self.classifier = nn.Sequential(
            nn.LayerNorm(representation_width),
            nn.Dropout(self.dropout),
            nn.Linear(representation_width, self.num_class),
        )
        self.last_routing_indices: List[torch.Tensor] = []

    def _apply_padding_mask(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if padding_mask is None or padding_mask.ndim != 2:
            return x
        if padding_mask.size(0) != x.size(0) or padding_mask.size(1) != x.size(1):
            return x
        return x * padding_mask.to(dtype=x.dtype).unsqueeze(-1)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: Optional[torch.Tensor] = None,
        x_dec: Optional[torch.Tensor] = None,
        x_mark_dec: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x_enc.ndim != 3:
            raise ValueError("ClinStab expects input with shape [batch, time, channels]")
        if x_enc.size(1) < 1:
            raise ValueError("ClinStab requires at least one time step")
        if x_enc.size(-1) != self.enc_in:
            raise ValueError(
                "ClinStab received {} channels, but enc_in is {}".format(
                    x_enc.size(-1), self.enc_in
                )
            )
        x_enc = self._apply_padding_mask(x_enc, x_mark_enc)
        representations = []
        routing = []
        for embedding, layers in zip(self.patch_embeddings, self.scdm_layers):
            features = embedding(x_enc)
            for layer in layers:
                features, indices = layer(features)
                routing.append(indices)
            # Pool only after channel-preserving fusion.  This is equivalent to
            # concatenating the refined granularity representations before the
            # classification head while keeping memory bounded for long windows.
            representations.append(features.mean(dim=(1, 2)))
        self.last_routing_indices = [item.detach() for item in routing]
        representation = torch.cat(representations, dim=-1)
        return self.classifier(representation)

    def routing_indices(self) -> List[torch.Tensor]:
        """Return detached TopK channel indices from the latest forward pass."""

        return list(self.last_routing_indices)
