# ------------------------------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/criterion.py
# Modified by Zuyao You (https://github.com/geshang777)
# ------------------------------------------------------------------------------------------------------
"""
MaskFormer criterion.
"""
import torch
import torch.fft as fft
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list, _max_by_axis
from ..utils import box_ops


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(dice_loss)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(sigmoid_ce_loss)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """Hungarian-matched class, box, mask, and boundary-phase supervision."""

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        eos_coef,
        losses,
        num_points,
        oversample_ratio,
        importance_sample_ratio,
        baphf_highpass=0.25,
        baphf_band_kernel=3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.baphf_highpass = baphf_highpass
        self.baphf_band_kernel = baphf_band_kernel

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """

        assert "pred_boxes" in outputs

        if indices is None or len(targets) == 0:
            loss = outputs["pred_boxes"].sum() * 0.0
            losses = {"loss_bbox_0": loss, "loss_giou_0": loss}
            return losses

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"]
        src_boxes = src_boxes[src_idx].sigmoid()

        target_boxes = [t["boxes"] for t in targets]
        max_size = _max_by_axis([list(box.shape) for box in target_boxes])
        max_size = [len(target_boxes)] + max_size
        empty_boxes = torch.zeros(max_size).to(src_boxes.device)
        for idx, tar_box in enumerate(target_boxes):
            empty_boxes[idx, : tar_box.shape[0], :] = tar_box
        target_boxes = empty_boxes[tgt_idx]

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes), box_ops.box_cxcywh_to_xyxy(target_boxes)
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: binary cross-entropy and Dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # Padding is included, matching the released training objective.
        target_masks, _ = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def loss_bap_hf(self, outputs, targets, indices, num_masks):
        """Align high-frequency phases inside the ground-truth boundary band.

        The normalization is over frequency bins, as in the retained training
        implementation; matched foreground/background regions are summed.
        """
        hp = float(self.baphf_highpass)

        if len(targets) == 0:
            zero = outputs["pred_masks"].sum() * 0.0
            return {"loss_bap_hf": zero}

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"][src_idx][:, None]  # (N,1,Hs,Ws)

        masks = [t["masks"] for t in targets]
        tgt_masks, _ = nested_tensor_from_tensor_list(masks).decompose()
        tgt_masks = tgt_masks.to(src_masks)
        tgt_masks = tgt_masks[tgt_idx][:, None]  # (N,1,Ht,Wt)

        # upsample preds to GT size
        src_prob = F.interpolate(
            src_masks, size=tgt_masks.shape[-2:], mode="bilinear", align_corners=False
        ).sigmoid()

        # Morphological boundary band, smoothed with average pooling.
        with torch.no_grad():
            k = max(1, int(self.baphf_band_kernel))
            pad = k // 2
            max_m = F.max_pool2d(tgt_masks, kernel_size=k, stride=1, padding=pad)
            min_m = -F.max_pool2d(-tgt_masks, kernel_size=k, stride=1, padding=pad)
            boundary = (max_m - min_m).clamp(min=0.0, max=1.0)
            band = F.avg_pool2d(boundary, kernel_size=k, stride=1, padding=pad)

        if band.sum() < 1:
            return {"loss_bap_hf": outputs["pred_masks"].sum() * 0.0}

        # apply boundary mask in spatial domain
        src_b = (src_prob * band).float()
        tgt_b = (tgt_masks * band).float()

        # FFT (complex)
        f_pred = fft.fftshift(fft.fft2(src_b, dim=(-2, -1)), dim=(-2, -1))
        f_gt = fft.fftshift(fft.fft2(tgt_b, dim=(-2, -1)), dim=(-2, -1))

        eps = 1e-6
        # high-pass radial mask (normalized radius in [0, ~1])
        _, _, H, W = f_gt.shape
        yy, xx = torch.meshgrid(
            torch.arange(H, device=f_gt.device),
            torch.arange(W, device=f_gt.device),
            indexing="ij",
        )
        yy = yy.float() - H / 2
        xx = xx.float() - W / 2
        rr = torch.sqrt(xx ** 2 + yy ** 2) / (0.5 * min(H, W) + eps)
        hp_mask = (rr >= hp).float().view(1, 1, H, W)

        # Per-frequency phase cosine penalty.
        p_pred = f_pred / (torch.abs(f_pred) + eps)
        p_gt = f_gt / (torch.abs(f_gt) + eps)
        cos = torch.real(p_pred * torch.conj(p_gt)).clamp(min=-1.0, max=1.0)
        num = ((1.0 - cos) * hp_mask).sum()
        den = hp_mask.sum() + eps
        loss = num / den
        return {"loss_bap_hf": loss}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            "labels": self.loss_labels,
            "masks": self.loss_masks,
            "boxes": self.loss_boxes,
            "bap_hf": self.loss_bap_hf,
        }
        if loss not in loss_map:
            raise ValueError(f"Unsupported loss: {loss}")
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == "bap_hf":
                        # BaP-HF supervises only the final decoder prediction.
                        continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
