"""XAI Evaluation Metrics Module.

This module provides objective, quantitative evaluation criteria for Explainable AI (XAI)
methods in Convolutional Neural Networks, covering:
1. Faithfulness / Fidelity: Deletion Metric and Insertion Metric (Petsiuk et al., 2018)
2. Localization & Pointing: Pointing Game Hit & Energy in Bounding Box (Zhang et al., 2018)
3. Sparsity & Complexity: Gini Index and Shannon Entropy of Attribution Maps
4. Model Sensitivity & Sanity Checks: Cascading Parameter Randomization (Adebayo et al., 2018)
"""

from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def normalize_attribution(attribution_map: np.ndarray) -> np.ndarray:
    """Normalize attribution map to [0, 1] range."""
    attr = np.maximum(attribution_map, 0)
    denom = np.max(attr) - np.min(attr)
    if denom > 1e-8:
        attr = (attr - np.min(attr)) / denom
    else:
        attr = np.zeros_like(attr)
    return attr


def compute_deletion_curve(
    model: nn.Module,
    input_tensor: torch.Tensor,
    attribution_map: np.ndarray,
    target_class_idx: int,
    steps: int = 20,
    device: torch.device | None = None,
    replacement: str = "constant",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute Deletion Curve and Area Under Curve (AUC).

    Higher faithfulness corresponds to a LOWER Deletion AUC (rapid drop in confidence
    when most important pixels are masked).

    Args:
        model: Evaluated PyTorch classification model.
        input_tensor: Tensor of shape (1, 3, H, W) normalized for the model.
        attribution_map: 2D array of shape (H, W) or (orig_H, orig_W).
        target_class_idx: Target ImageNet class index.
        steps: Number of removal increments (e.g., 20 steps -> 5% increments).
        device: Device to run inference on.
        replacement: 'constant' (zeros) or 'mean'.

    Returns:
        removed_fractions: Array of removed percentage [0.0, ..., 1.0]
        scores: Predicted class probabilities at each step
        auc_score: Area under the deletion curve (lower is more faithful)
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    input_tensor = input_tensor.to(device)
    _, _, h, w = input_tensor.shape

    if attribution_map.shape != (h, w):
        attr_pil = Image.fromarray(attribution_map.astype(np.float32))
        attr_pil = attr_pil.resize((w, h), Image.Resampling.BILINEAR)
        attr_map = np.array(attr_pil)
    else:
        attr_map = attribution_map

    flat_attr = attr_map.flatten()
    sorted_indices = np.argsort(-flat_attr)

    total_pixels = h * w
    step_size = total_pixels // steps

    if replacement == "mean":
        mean_val = input_tensor.mean(dim=(2, 3), keepdim=True)
        baseline = mean_val.expand_as(input_tensor)
    else:
        baseline = torch.zeros_like(input_tensor)

    scores = []
    fractions = []
    modified_tensor = input_tensor.clone()

    with torch.no_grad():
        init_prob = torch.softmax(model(modified_tensor), dim=1)[
            0, target_class_idx
        ].item()
        scores.append(init_prob)
        fractions.append(0.0)

        flat_modified = modified_tensor.view(1, 3, -1)
        flat_baseline = baseline.view(1, 3, -1)

        for s in range(1, steps + 1):
            idx_end = min(s * step_size, total_pixels)
            chunk = sorted_indices[(s - 1) * step_size : idx_end]
            flat_modified[:, :, chunk] = flat_baseline[:, :, chunk]

            cur_prob = torch.softmax(model(modified_tensor), dim=1)[
                0, target_class_idx
            ].item()
            scores.append(cur_prob)
            fractions.append(idx_end / total_pixels)

    scores_arr = np.array(scores)
    fractions_arr = np.array(fractions)
    auc_score = float(np.trapezoid(scores_arr, fractions_arr))

    return fractions_arr, scores_arr, auc_score


def compute_insertion_curve(
    model: nn.Module,
    input_tensor: torch.Tensor,
    attribution_map: np.ndarray,
    target_class_idx: int,
    steps: int = 20,
    device: torch.device | None = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute Insertion Curve and Area Under Curve (AUC).

    Higher faithfulness corresponds to a HIGHER Insertion AUC (rapid recovery of
    confidence when important pixels are restored from baseline).
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    input_tensor = input_tensor.to(device)
    _, _, h, w = input_tensor.shape

    if attribution_map.shape != (h, w):
        attr_pil = Image.fromarray(attribution_map.astype(np.float32))
        attr_pil = attr_pil.resize((w, h), Image.Resampling.BILINEAR)
        attr_map = np.array(attr_pil)
    else:
        attr_map = attribution_map

    flat_attr = attr_map.flatten()
    sorted_indices = np.argsort(-flat_attr)

    total_pixels = h * w
    step_size = total_pixels // steps

    baseline = torch.zeros_like(input_tensor)
    canvas = baseline.clone()

    scores = []
    fractions = []

    with torch.no_grad():
        init_prob = torch.softmax(model(canvas), dim=1)[0, target_class_idx].item()
        scores.append(init_prob)
        fractions.append(0.0)

        flat_canvas = canvas.view(1, 3, -1)
        flat_input = input_tensor.view(1, 3, -1)

        for s in range(1, steps + 1):
            idx_end = min(s * step_size, total_pixels)
            chunk = sorted_indices[(s - 1) * step_size : idx_end]
            flat_canvas[:, :, chunk] = flat_input[:, :, chunk]

            cur_prob = torch.softmax(model(canvas), dim=1)[0, target_class_idx].item()
            scores.append(cur_prob)
            fractions.append(idx_end / total_pixels)

    scores_arr = np.array(scores)
    fractions_arr = np.array(fractions)
    auc_score = float(np.trapezoid(scores_arr, fractions_arr))

    return fractions_arr, scores_arr, auc_score


def compute_pointing_game_hit(
    attribution_map: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> bool:
    """Evaluate whether the maximum attribution point falls within ground truth bbox.

    Args:
        attribution_map: 2D heatmap array.
        bbox: (xmin, ymin, xmax, ymax) coordinates.

    Returns:
        bool: True if argmax(attribution) is inside bbox, False otherwise.
    """
    max_idx = np.unravel_index(np.argmax(attribution_map), attribution_map.shape)
    y_max, x_max = max_idx[0], max_idx[1]
    xmin, ymin, xmax, ymax = bbox
    return bool((xmin <= x_max <= xmax) and (ymin <= y_max <= ymax))


def compute_energy_in_bbox(
    attribution_map: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> float:
    """Calculate the fraction of attribution energy located inside a bounding box."""
    norm_attr = normalize_attribution(attribution_map)
    total_energy = float(np.sum(norm_attr))
    if total_energy <= 1e-12:
        return 0.0

    xmin, ymin, xmax, ymax = bbox
    xmin = max(0, int(xmin))
    ymin = max(0, int(ymin))
    xmax = min(norm_attr.shape[1], int(xmax))
    ymax = min(norm_attr.shape[0], int(ymax))

    bbox_energy = float(np.sum(norm_attr[ymin:ymax, xmin:xmax]))
    return bbox_energy / total_energy


def compute_sparsity_gini(attribution_map: np.ndarray) -> float:
    """Compute Gini coefficient of the attribution map to measure sparsity/complexity.

    Higher Gini (closer to 1.0) means explanation is sparse and concentrated on few pixels.
    Lower Gini (closer to 0.0) means explanation is uniformly dispersed (noisy).
    """
    norm_attr = normalize_attribution(attribution_map).flatten()
    norm_attr = norm_attr + 1e-12
    sorted_attr = np.sort(norm_attr)
    n = len(sorted_attr)
    index = np.arange(1, n + 1)
    gini = (2.0 * np.sum(index * sorted_attr)) / (n * np.sum(sorted_attr)) - (
        n + 1.0
    ) / n
    return float(np.clip(gini, 0.0, 1.0))


def compute_normalized_entropy(attribution_map: np.ndarray) -> float:
    """Compute normalized Shannon entropy of the attribution map.

    Values in [0, 1]. Lower entropy means focused explanation; 1.0 means maximal diffusion.
    """
    norm_attr = normalize_attribution(attribution_map).flatten()
    total = np.sum(norm_attr)
    if total <= 1e-12:
        return 1.0
    prob = norm_attr / total
    prob = prob[prob > 0]
    entropy = -np.sum(prob * np.log2(prob))
    max_entropy = np.log2(len(norm_attr))
    return float(entropy / max_entropy) if max_entropy > 0 else 1.0


def randomize_layers_cascading(
    model: nn.Module,
    layer_names_cascading: List[str],
) -> None:
    """Cascading Parameter Randomization Sanity Check (Adebayo et al., 2018).

    Resets weights of model layers sequentially from top (classifier) down to lower blocks.
    """
    for name, module in model.named_modules():
        if name in layer_names_cascading:
            reset_parameters = getattr(module, "reset_parameters", None)
            if callable(reset_parameters):
                reset_parameters()
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)
