from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - torch is expected in training env
    torch = None


@dataclass
class ErosionSegmentResult:
    mask: np.ndarray
    pruned_skeleton: np.ndarray
    branch_mask: np.ndarray
    segment_labels: np.ndarray
    eroded_mask: np.ndarray
    disappearing_segment_ids: list[int]
    segment_stats: list[dict]


def _degrees(skeleton: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    return ndi.convolve(skeleton.astype(np.uint8), kernel, mode="constant", cval=0) - skeleton.astype(np.uint8)


def _prune_short_spurs(skeleton: np.ndarray, spur_max_len: int) -> np.ndarray:
    pruned = skeleton.copy()
    neighbors8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    h, w = pruned.shape

    for _ in range(10):
        deg = _degrees(pruned)
        endpoints = np.argwhere(pruned & (deg == 1))
        removed = False
        for y0, x0 in endpoints:
            if not pruned[y0, x0]:
                continue

            path = [(y0, x0)]
            prev = None
            y, x = y0, x0
            for _step in range(spur_max_len + 2):
                neigh = []
                for dy, dx in neighbors8:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and pruned[ny, nx] and (prev is None or (ny, nx) != prev):
                        neigh.append((ny, nx))

                if len(neigh) == 0 or len(neigh) >= 2:
                    break

                prev = (y, x)
                y, x = neigh[0]
                path.append((y, x))
                deg_now = _degrees(pruned)[y, x]
                if deg_now >= 3 or (deg_now == 1 and (y, x) != (y0, x0)):
                    break

            end_deg = _degrees(pruned)[y, x]
            if len(path) <= spur_max_len and end_deg >= 3:
                for py, px in path[:-1]:
                    pruned[py, px] = False
                removed = True

        if not removed:
            break

    return pruned


def extract_path_segment_labels(
    mask: np.ndarray,
    spur_max_len: int = 10,
    min_segment_len: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split a binary vessel mask into path segments.

    A segment is defined as one skeleton path between branch/end nodes.
    The returned labels are expanded from the skeleton back to the full vessel area.
    """
    vessel = mask.astype(bool)
    h, w = vessel.shape
    neighbors8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    structure = np.ones((3, 3), dtype=np.uint8)

    skeleton = skeletonize(vessel)
    pruned = _prune_short_spurs(skeleton, spur_max_len=spur_max_len)
    pr_deg = _degrees(pruned)
    branch_mask = pruned & (pr_deg >= 3)

    edge_mask = pruned & (~branch_mask)
    edge_labels, n_edges = ndi.label(edge_mask, structure=structure)
    edge_sizes = ndi.sum(edge_mask, edge_labels, index=np.arange(1, n_edges + 1))

    filtered = np.zeros_like(edge_labels)
    next_id = 0
    for old_id, size in enumerate(edge_sizes, start=1):
        if size < min_segment_len:
            continue
        next_id += 1
        filtered[edge_labels == old_id] = next_id
    edge_labels = filtered

    labels = np.zeros((h, w), dtype=np.int32)
    queue: deque[tuple[int, int, int]] = deque()
    for y, x in np.argwhere(edge_labels > 0):
        seg_id = int(edge_labels[y, x])
        labels[y, x] = seg_id
        queue.append((y, x, seg_id))

    while queue:
        y, x, seg_id = queue.popleft()
        for dy, dx in neighbors8:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and vessel[ny, nx] and labels[ny, nx] == 0 and not branch_mask[ny, nx]:
                labels[ny, nx] = seg_id
                queue.append((ny, nx, seg_id))

    unassigned = vessel & (labels == 0)
    for _ in range(20):
        if not unassigned.any():
            break
        changed = False
        ys, xs = np.where(unassigned)
        for y, x in zip(ys, xs):
            neigh_labels = []
            for dy, dx in neighbors8:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and labels[ny, nx] > 0:
                    neigh_labels.append(labels[ny, nx])
            if neigh_labels:
                values, counts = np.unique(neigh_labels, return_counts=True)
                labels[y, x] = int(values[np.argmax(counts)])
                changed = True
        unassigned = vessel & (labels == 0)
        if not changed:
            break

    if (vessel & (labels == 0)).any() and (labels > 0).any():
        _, indices = ndi.distance_transform_edt(labels == 0, return_indices=True)
        yy, xx = np.where(vessel & (labels == 0))
        labels[yy, xx] = labels[indices[0, yy, xx], indices[1, yy, xx]]

    return labels, pruned, branch_mask


def find_segments_disappeared_by_erosion(
    mask: np.ndarray,
    erosion_kernel_size: int = 3,
    spur_max_len: int = 10,
    min_segment_len: int = 6,
) -> ErosionSegmentResult:
    """
    Keep the segment construction logic and mark segments that disappear completely after erosion.
    """
    vessel = mask.astype(bool)
    segment_labels, pruned_skeleton, branch_mask = extract_path_segment_labels(
        vessel,
        spur_max_len=spur_max_len,
        min_segment_len=min_segment_len,
    )

    structure = np.ones((erosion_kernel_size, erosion_kernel_size), dtype=bool)
    eroded = ndi.binary_erosion(vessel, structure=structure)

    segment_stats: list[dict] = []
    disappearing_ids: list[int] = []
    for seg_id in sorted(int(x) for x in np.unique(segment_labels) if x > 0):
        segment = segment_labels == seg_id
        total = int(segment.sum())
        survived = int((segment & eroded).sum())
        vanished = total - survived
        vanished_ratio = vanished / total if total > 0 else 0.0
        stat = {
            "segment_id": seg_id,
            "total_pixels": total,
            "survived_pixels": survived,
            "vanished_pixels": vanished,
            "vanished_ratio": vanished_ratio,
            "disappeared": survived == 0,
        }
        segment_stats.append(stat)
        if survived == 0:
            disappearing_ids.append(seg_id)

    return ErosionSegmentResult(
        mask=vessel,
        pruned_skeleton=pruned_skeleton,
        branch_mask=branch_mask,
        segment_labels=segment_labels,
        eroded_mask=eroded,
        disappearing_segment_ids=disappearing_ids,
        segment_stats=segment_stats,
    )


def render_disappearing_segments(
    result: ErosionSegmentResult,
    colors: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return:
    - colored segments whose full branch disappears after erosion
    - reference image with exact eroded-away region highlighted
    - 4-panel image: original | eroded | colored-segments | reference
    """
    if colors is None:
        colors = np.array(
            [
                [230, 57, 70],
                [29, 53, 87],
                [69, 123, 157],
                [168, 218, 220],
                [42, 157, 143],
                [233, 196, 106],
                [244, 162, 97],
                [231, 111, 81],
                [106, 76, 147],
                [17, 138, 178],
                [6, 214, 160],
                [255, 209, 102],
                [239, 71, 111],
                [255, 0, 110],
                [58, 134, 255],
                [251, 133, 0],
            ],
            dtype=np.uint8,
        )

    h, w = result.mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    colored[result.mask] = [65, 65, 65]
    for idx, seg_id in enumerate(result.disappearing_segment_ids):
        colored[result.segment_labels == seg_id] = colors[idx % len(colors)]

    reference = colored.copy()
    reference[result.mask & (~result.eroded_mask)] = [0, 255, 255]

    original = np.zeros((h, w, 3), dtype=np.uint8)
    original[result.mask] = [255, 255, 255]
    eroded = np.zeros((h, w, 3), dtype=np.uint8)
    eroded[result.eroded_mask] = [255, 255, 255]
    panel = np.concatenate([original, eroded, colored, reference], axis=1)

    return colored, reference, panel


def build_disappearing_segment_region(
    mask: np.ndarray,
    erosion_kernel_size: int = 3,
    spur_max_len: int = 10,
    min_segment_len: int = 6,
) -> np.ndarray:
    """
    Build a binary region mask containing full path segments that disappear after erosion.

    This is intended to be used as a loss region.
    """
    result = find_segments_disappeared_by_erosion(
        mask=mask,
        erosion_kernel_size=erosion_kernel_size,
        spur_max_len=spur_max_len,
        min_segment_len=min_segment_len,
    )
    region = np.zeros_like(result.mask, dtype=np.uint8)
    for seg_id in result.disappearing_segment_ids:
        region[result.segment_labels == seg_id] = 1
    return region


def build_pred_disappearing_segment_region(
    pred: np.ndarray,
    threshold: float = 0.5,
    erosion_kernel_size: int = 3,
    spur_max_len: int = 10,
    min_segment_len: int = 6,
) -> np.ndarray:
    """
    Convert a prediction map into a binary loss region mask.

    `pred` can be either probabilities/logits already thresholded to [0,1] or a binary mask.
    """
    pred_arr = np.asarray(pred)
    if pred_arr.dtype != np.bool_:
        pred_arr = pred_arr > threshold
    return build_disappearing_segment_region(
        pred_arr,
        erosion_kernel_size=erosion_kernel_size,
        spur_max_len=spur_max_len,
        min_segment_len=min_segment_len,
    )


def build_pred_disappearing_segment_region_torch(
    pred: "torch.Tensor",
    threshold: float = 0.5,
    erosion_kernel_size: int = 3,
    spur_max_len: int = 10,
    min_segment_len: int = 6,
) -> "torch.Tensor":
    """
    Torch wrapper for training-time use.

    Input:
    - binary segmentation probability/logit tensor with shape [B,H,W] or [B,1,H,W]

    Output:
    - binary region mask tensor with the same spatial size and shape [B,1,H,W]
    """
    if torch is None:
        raise ImportError("torch is required for build_pred_disappearing_segment_region_torch")

    pred_tensor = pred.detach()
    if pred_tensor.ndim == 4 and pred_tensor.shape[1] == 1:
        pred_tensor = pred_tensor[:, 0]
    elif pred_tensor.ndim != 3:
        raise ValueError(f"Expected pred shape [B,H,W] or [B,1,H,W], got {tuple(pred.shape)}")

    batch_regions = []
    pred_np = pred_tensor.float().cpu().numpy()
    for sample in pred_np:
        region = build_pred_disappearing_segment_region(
            sample,
            threshold=threshold,
            erosion_kernel_size=erosion_kernel_size,
            spur_max_len=spur_max_len,
            min_segment_len=min_segment_len,
        )
        batch_regions.append(region.astype(np.float32))

    region_tensor = torch.from_numpy(np.stack(batch_regions, axis=0)).to(pred.device)
    return region_tensor.unsqueeze(1)


def _safe_region_bce_with_logits(
    logits: "torch.Tensor",
    targets: "torch.Tensor",
    region_mask: "torch.Tensor",
    eps: float = 1e-8,
) -> "torch.Tensor":
    pixel_bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weighted = pixel_bce * region_mask
    denom = region_mask.sum().clamp_min(eps)
    return weighted.sum() / denom


def loss1_gt_disappearing_segments(
    logits: "torch.Tensor",
    targets: "torch.Tensor",
    erosion_kernel_size: int = 3,
    spur_max_len: int = 10,
    min_segment_len: int = 6,
    eps: float = 1e-8,
    return_region: bool = False,
):
    """
    Loss 1:
    Build disappearing-segment regions from GT, then compute region-weighted BCE.

    This is a supervised thin/fragile branch emphasis term.
    """
    if torch is None:
        raise ImportError("torch is required for loss1_gt_disappearing_segments")

    if targets.ndim == 3:
        target_tensor = targets.unsqueeze(1).float()
    elif targets.ndim == 4 and targets.shape[1] == 1:
        target_tensor = targets.float()
    else:
        raise ValueError(f"Expected targets shape [B,H,W] or [B,1,H,W], got {tuple(targets.shape)}")

    region_mask = build_pred_disappearing_segment_region_torch(
        target_tensor,
        threshold=0.5,
        erosion_kernel_size=erosion_kernel_size,
        spur_max_len=spur_max_len,
        min_segment_len=min_segment_len,
    ).to(logits.device)

    loss = _safe_region_bce_with_logits(logits, target_tensor.to(logits.device), region_mask, eps=eps)
    if return_region:
        return loss, region_mask
    return loss


def loss2_pred_disappearing_segments(
    logits: "torch.Tensor",
    targets: "torch.Tensor",
    threshold: float = 0.5,
    erosion_kernel_size: int = 3,
    spur_max_len: int = 10,
    min_segment_len: int = 6,
    eps: float = 1e-8,
    return_region: bool = False,
):
    """
    Loss 2:
    Build disappearing-segment regions from the current prediction, then compute region-weighted BCE.

    The region extraction is detached from gradients and acts as a dynamic hard-region miner.
    """
    if torch is None:
        raise ImportError("torch is required for loss2_pred_disappearing_segments")

    if targets.ndim == 3:
        target_tensor = targets.unsqueeze(1).float()
    elif targets.ndim == 4 and targets.shape[1] == 1:
        target_tensor = targets.float()
    else:
        raise ValueError(f"Expected targets shape [B,H,W] or [B,1,H,W], got {tuple(targets.shape)}")

    pred_prob = torch.sigmoid(logits.detach())
    region_mask = build_pred_disappearing_segment_region_torch(
        pred_prob,
        threshold=threshold,
        erosion_kernel_size=erosion_kernel_size,
        spur_max_len=spur_max_len,
        min_segment_len=min_segment_len,
    ).to(logits.device)

    loss = _safe_region_bce_with_logits(logits, target_tensor.to(logits.device), region_mask, eps=eps)
    if return_region:
        return loss, region_mask
    return loss
