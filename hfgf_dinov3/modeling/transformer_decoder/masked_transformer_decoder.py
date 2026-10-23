# Copyright (c) Facebook, Inc. and its affiliates.
# Adapted from Mask2Former and FOCUS:
# https://github.com/facebookresearch/Mask2Former
# Original FOCUS modifications by Zuyao You (https://github.com/geshang777).
"""Foreground/background masked transformer decoder used by HFGF-DINOv3."""

from typing import Optional

import fvcore.nn.weight_init as weight_init
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d
from detectron2.utils.registry import Registry

from .position_encoding import PositionEmbeddingSine
from .transformer import _get_activation_fn

TRANSFORMER_DECODER_REGISTRY = Registry("TRANSFORMER_MODULE")


def build_transformer_decoder(cfg, in_channels, mask_classification=True):
    name = cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME
    return TRANSFORMER_DECODER_REGISTRY.get(name)(cfg, in_channels, mask_classification)


class SelfAttentionLayer(nn.Module):
    def __init__(
        self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask, tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask, tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):
    def __init__(
        self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False
    ):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = self.with_pos_embed(tgt, query_pos)
        k = self.with_pos_embed(memory, pos)
        attn_mask = memory_mask
        query_length = q.shape[0]
        if attn_mask is not None and attn_mask.dim() == 3:
            if attn_mask.shape[1] > query_length:
                attn_mask = attn_mask[:, :query_length, :]
            elif attn_mask.shape[1] < query_length:
                pad_len = query_length - attn_mask.shape[1]
                pad = torch.zeros(
                    (attn_mask.shape[0], pad_len, attn_mask.shape[2]),
                    device=attn_mask.device,
                    dtype=attn_mask.dtype,
                )
                attn_mask = torch.cat([attn_mask, pad], dim=1)
        tgt2 = self.multihead_attn(
            query=q,
            key=k,
            value=memory,
            attn_mask=attn_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm(tgt)
        q = self.with_pos_embed(tgt2, query_pos)
        k = self.with_pos_embed(memory, pos)
        attn_mask = memory_mask
        query_length = q.shape[0]
        if attn_mask is not None and attn_mask.dim() == 3:
            if attn_mask.shape[1] > query_length:
                attn_mask = attn_mask[:, :query_length, :]
            elif attn_mask.shape[1] < query_length:
                pad_len = query_length - attn_mask.shape[1]
                pad = torch.zeros(
                    (attn_mask.shape[0], pad_len, attn_mask.shape[2]),
                    device=attn_mask.device,
                    dtype=attn_mask.dtype,
                )
                attn_mask = torch.cat([attn_mask, pad], dim=1)
        tgt2 = self.multihead_attn(
            query=q,
            key=k,
            value=memory,
            attn_mask=attn_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos
            )
        return self.forward_post(
            tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos
        )


class FFNLayer(nn.Module):
    def __init__(
        self,
        d_model,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@TRANSFORMER_DECODER_REGISTRY.register()
class MultiScaleMaskedTransformerDecoder(nn.Module):
    """Decode two learned queries over three pixel-decoder feature scales."""

    _version = 2

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        use_feature_mask_attn: bool = True,
    ):
        super().__init__()
        if not mask_classification:
            raise ValueError("HFGF-DINOv3 requires mask classification.")
        self.mask_classification = mask_classification
        self.pe_layer = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    hidden_dim, nheads, dropout=0.0, normalize_before=pre_norm
                )
            )
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    hidden_dim, nheads, dropout=0.0, normalize_before=pre_norm
                )
            )
            self.transformer_ffn_layers.append(
                FFNLayer(
                    hidden_dim, dim_feedforward, dropout=0.0, normalize_before=pre_norm
                )
            )
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.num_queries = num_queries
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.use_feature_mask_attn = use_feature_mask_attn

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        decoder = cfg.MODEL.MASK_FORMER
        if decoder.DEC_LAYERS < 1:
            raise ValueError(
                "DEC_LAYERS must include at least the initial prediction head."
            )
        return {
            "in_channels": in_channels,
            "mask_classification": mask_classification,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "hidden_dim": decoder.HIDDEN_DIM,
            "num_queries": decoder.NUM_OBJECT_QUERIES,
            "nheads": decoder.NHEADS,
            "dim_feedforward": decoder.DIM_FEEDFORWARD,
            "dec_layers": decoder.DEC_LAYERS - 1,
            "pre_norm": decoder.PRE_NORM,
            "enforce_input_project": decoder.ENFORCE_INPUT_PROJ,
            "mask_dim": cfg.MODEL.SEM_SEG_HEAD.MASK_DIM,
            "use_feature_mask_attn": decoder.get("USE_FEATURE_MASK_ATTN", True),
        }

    def forward(self, features, mask_features, training_context=None):
        """Return class, mask and box predictions, plus intermediate supervision.

        During training, ``training_context['feature_mask']`` contains the
        backbone PCA foreground prior used by the first attention layer.
        """
        if len(features) != self.num_feature_levels:
            raise ValueError("The decoder expects exactly three feature scales.")
        src, pos, sizes = [], [], []
        for index, feature in enumerate(features):
            sizes.append(feature.shape[-2:])
            pos.append(self.pe_layer(feature, None).flatten(2))
            src.append(
                self.input_proj[index](feature).flatten(2)
                + self.level_embed.weight[index][None, :, None]
            )
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        batch_size = src[0].shape[1]
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, batch_size, 1)
        classes, masks, boxes = [], [], []
        (
            outputs_class,
            outputs_mask,
            attn_mask,
            outputs_bbox,
        ) = self.forward_prediction_heads(output, mask_features, sizes[0])
        prior = (
            None if training_context is None else training_context.get("feature_mask")
        )
        if self.training and self.use_feature_mask_attn and prior is not None:
            _, labels = torch.max(outputs_class, dim=2)
            if not torch.is_tensor(prior):
                prior = torch.from_numpy(prior)
            feature_mask = prior.to(dtype=torch.float32, device="cpu")
            attn_mask = F.interpolate(
                feature_mask.unsqueeze(1),
                size=sizes[0],
                mode="bilinear",
                align_corners=False,
            )
            # Keep the paper's training graph, including normalization and
            # foreground/background ordering, for checkpoint reproducibility.
            attn_mask = (attn_mask - attn_mask.min()) / (
                attn_mask.max() - attn_mask.min()
            )
            reversed_mask = 1 - attn_mask
            attn_mask = (
                torch.cat((attn_mask, reversed_mask), dim=1)
                if labels[0][0] == 0
                else torch.cat((reversed_mask, attn_mask), dim=1)
            )
            attn_mask = (
                attn_mask.sigmoid()
                .flatten(2)
                .unsqueeze(1)
                .repeat(1, self.num_heads, 1, 1)
                .flatten(0, 1)
                < 0.5
            ).bool()
            attn_mask = attn_mask.to(device=outputs_mask.device).detach()
        classes.append(outputs_class)
        masks.append(outputs_mask)
        boxes.append(outputs_bbox)

        for index in range(self.num_layers):
            level = index % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            output = self.transformer_cross_attention_layers[index](
                output,
                src[level],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,
                pos=pos[level],
                query_pos=query_embed,
            )
            output = self.transformer_self_attention_layers[index](
                output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_embed
            )
            output = self.transformer_ffn_layers[index](output)
            (
                outputs_class,
                outputs_mask,
                attn_mask,
                outputs_bbox,
            ) = self.forward_prediction_heads(
                output, mask_features, sizes[(index + 1) % self.num_feature_levels]
            )
            classes.append(outputs_class)
            masks.append(outputs_mask)
            boxes.append(outputs_bbox)

        return {
            "pred_logits": classes[-1],
            "pred_masks": masks[-1],
            "pred_boxes": boxes[-1],
            "mask_features": mask_features,
            "aux_outputs": self._set_aux_loss(classes, masks, boxes),
        }

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output).transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        attn_mask = F.interpolate(
            outputs_mask,
            size=attn_mask_target_size,
            mode="bilinear",
            align_corners=False,
        )
        # True entries block attention at the next feature scale.
        attn_mask = (
            (
                attn_mask.sigmoid()
                .flatten(2)
                .unsqueeze(1)
                .repeat(1, self.num_heads, 1, 1)
                .flatten(0, 1)
                < 0.5
            )
            .bool()
            .detach()
        )
        outputs_bbox = self.bbox_embed(decoder_output)
        return outputs_class, outputs_mask, attn_mask, outputs_bbox

    @torch.jit.unused
    def _set_aux_loss(self, classes, masks, boxes):
        return [
            {"pred_logits": logits, "pred_masks": mask, "pred_boxes": box}
            for logits, mask, box in zip(classes[:-1], masks[:-1], boxes[:-1])
        ]
