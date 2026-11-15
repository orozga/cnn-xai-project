import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

AVAILABLE_METHODS: tuple[str, ...] = (
    "saliency",
    "gradcam",
    "gradcam_pp",
    "scorecam",
)

METHOD_LABELS: dict[str, str] = {
    "saliency": "Saliency Maps",
    "gradcam": "Grad-CAM",
    "gradcam_pp": "Grad-CAM++",
    "scorecam": "Score-CAM",
}


def _unwrap_model(model_or_wrapper: Any) -> nn.Module:
    return (
        model_or_wrapper.model
        if hasattr(model_or_wrapper, "model")
        else model_or_wrapper
    )


def _normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.nan_to_num(heatmap.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    min_val = float(np.min(heatmap))
    max_val = float(np.max(heatmap))
    if max_val - min_val > 1e-8:
        return (heatmap - min_val) / (max_val - min_val)
    return np.zeros_like(heatmap, dtype=np.float32)


def _extract_activations_and_gradients(
    model_or_wrapper: Any,
    input_batch: torch.Tensor,
    target_class_idx: int | None = None,
) -> tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
    """Run forward and backward pass, returning (logits, target_class_idx, activations, gradients)."""
    if hasattr(model_or_wrapper, "forward_and_backward"):
        return model_or_wrapper.forward_and_backward(
            input_batch, target_class_idx=target_class_idx
        )

    model: nn.Module = _unwrap_model(model_or_wrapper)
    model.eval()
    device = next(model.parameters()).device
    x = input_batch.to(device)

    captured: dict[str, torch.Tensor] = {}

    def fwd_hook(module, inp, out):
        captured["activations"] = out

    def bwd_hook(module, grad_in, grad_out):
        captured["gradients"] = grad_out[0]

    target_layer = model.layer4
    h_fwd = target_layer.register_forward_hook(fwd_hook)
    h_bwd = target_layer.register_full_backward_hook(bwd_hook)

    try:
        output = model(x)
        if target_class_idx is None:
            target_class_idx = int(output.argmax(dim=1).item())

        model.zero_grad()
        score = output[0, target_class_idx]
        score.backward()

        return (
            output.detach(),
            int(target_class_idx),
            captured["activations"].detach(),
            captured["gradients"].detach(),
        )
    finally:
        h_fwd.remove()
        h_bwd.remove()


def compute_saliency_map(
    model: Any,
    input_batch: torch.Tensor,
    target_class_idx: int | None = None,
) -> tuple[np.ndarray, int]:
    """Compute Vanilla Saliency Map"""
    raw_model = _unwrap_model(model)
    raw_model.eval()
    device = next(raw_model.parameters()).device

    input_tensor = input_batch.clone().detach().to(device)
    input_tensor.requires_grad_()
    output = raw_model(input_tensor)

    if target_class_idx is None:
        target_class_idx = int(output.argmax(dim=1).item())

    score = output[0, target_class_idx]
    raw_model.zero_grad()
    if input_tensor.grad is not None:
        input_tensor.grad.zero_()

    score.backward()

    assert input_tensor.grad is not None
    gradients = input_tensor.grad.data.abs()
    saliency, _ = torch.max(gradients, dim=1)
    saliency_np = _normalize_heatmap(saliency[0].cpu().numpy())

    return saliency_np, int(target_class_idx)


def compute_gradcam(
    model: Any,
    input_batch: torch.Tensor,
    target_class_idx: int | None = None,
) -> tuple[np.ndarray, int]:
    """Compute Grad-CAM heatmap for the target class."""
    _, target_class_idx, activations, gradients = _extract_activations_and_gradients(
        model, input_batch, target_class_idx=target_class_idx
    )
    _, _, h, w = input_batch.shape

    # Global Average Pooling of gradients: alpha_k^c = (1/Z) * sum_i sum_j d y^c / d A_{i,j}^k
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

    # L_{Grad-CAM}^c = ReLU(sum_k alpha_k^c * A^k)
    cam = torch.sum(weights * activations, dim=1, keepdim=True)
    cam = F.relu(cam)

    cam_upsampled = F.interpolate(
        cam, size=(h, w), mode="bilinear", align_corners=False
    )
    cam_np = _normalize_heatmap(cam_upsampled[0, 0].cpu().numpy())

    return cam_np, int(target_class_idx)


def compute_gradcam_pp(
    model: Any,
    input_batch: torch.Tensor,
    target_class_idx: int | None = None,
) -> tuple[np.ndarray, int]:
    """Compute Grad-CAM++ heatmap using pixel-wise higher-order gradient weighting."""
    _, target_class_idx, activations, gradients = _extract_activations_and_gradients(
        model, input_batch, target_class_idx=target_class_idx
    )
    _, _, h, w = input_batch.shape

    # Pixel-wise weighting coefficients alpha_{i,j}^{k,c}
    grad_2 = gradients.pow(2)
    grad_3 = gradients.pow(3)
    spatial_sum = torch.sum(activations * grad_3, dim=(2, 3), keepdim=True)
    denom = 2.0 * grad_2 + spatial_sum + 1e-8
    alpha = grad_2 / denom

    # w_k^c = sum_i sum_j alpha_{i,j}^{k,c} * ReLU(d y^c / d A_{i,j}^k)
    positive_gradients = F.relu(gradients)
    weights = torch.sum(alpha * positive_gradients, dim=(2, 3), keepdim=True)

    cam = torch.sum(weights * activations, dim=1, keepdim=True)
    cam = F.relu(cam)

    cam_upsampled = F.interpolate(
        cam, size=(h, w), mode="bilinear", align_corners=False
    )
    cam_np = _normalize_heatmap(cam_upsampled[0, 0].cpu().numpy())

    return cam_np, int(target_class_idx)


def compute_scorecam(
    model: Any,
    input_batch: torch.Tensor,
    target_class_idx: int | None = None,
    batch_size: int = 64,
    max_channels: int | None = None,
) -> tuple[np.ndarray, int]:
    """Compute gradient-free Score-CAM heatmap via Channel-wise Increase of Confidence (CIC)."""
    raw_model = _unwrap_model(model)
    raw_model.eval()
    device = next(raw_model.parameters()).device
    x = input_batch.to(device)
    _, _, h, w = x.shape

    captured: dict[str, torch.Tensor] = {}

    def fwd_hook(module, inp, out):
        captured["activations"] = out.detach()

    target_layer = raw_model.layer4
    h_fwd = target_layer.register_forward_hook(fwd_hook)

    try:
        with torch.no_grad():
            output = raw_model(x)
            if target_class_idx is None:
                target_class_idx = int(output.argmax(dim=1).item())

            activations = captured["activations"]  # (1, K, H_l, W_l)

            # Baseline image X_b (zeros)
            baseline = torch.zeros_like(x)
            base_probs = F.softmax(raw_model(baseline), dim=1)
            base_score = base_probs[0, target_class_idx]

            total_channels = activations.shape[1]
            if max_channels is not None and 0 < max_channels < total_channels:
                channel_means = activations[0].mean(dim=(1, 2))
                _, selected_indices = torch.topk(channel_means, k=max_channels)
                selected_activations = activations[:, selected_indices, :, :]
            else:
                selected_activations = activations

            num_selected = selected_activations.shape[1]
            cic_weights = torch.zeros(num_selected, device=device)

            # Upsample all selected activation maps: Up(A_l^k) -> (num_selected, 1, H, W)
            act_reshaped = selected_activations.transpose(0, 1)
            upsampled = F.interpolate(
                act_reshaped, size=(h, w), mode="bilinear", align_corners=False
            )

            # Normalize each channel map to [0, 1]: s(Up(A_l^k))
            flat = upsampled.view(num_selected, -1)
            mins = flat.min(dim=1, keepdim=True)[0].view(num_selected, 1, 1, 1)
            maxs = flat.max(dim=1, keepdim=True)[0].view(num_selected, 1, 1, 1)
            spans = maxs - mins
            valid_mask = (spans.view(-1) > 1e-8)
            norm_masks = torch.where(
                spans > 1e-8,
                (upsampled - mins) / (spans + 1e-8),
                torch.zeros_like(upsampled),
            )

            # Evaluate CIC in batches: f^c(X o s(Up(A_l^k))) - f^c(X_b)
            for start in range(0, num_selected, batch_size):
                end = min(start + batch_size, num_selected)
                batch_masks = norm_masks[start:end]
                masked_inputs = x * batch_masks
                logits = raw_model(masked_inputs)
                probs = F.softmax(logits, dim=1)[:, target_class_idx]
                cic_weights[start:end] = probs - base_score

            cic_weights = cic_weights * valid_mask.float()
            weights = cic_weights.view(1, num_selected, 1, 1)

            # L_{Score-CAM}^c = ReLU(sum_k alpha_k^c * A_l^k)
            cam = torch.sum(weights * selected_activations, dim=1, keepdim=True)
            cam = F.relu(cam)
            cam_upsampled = F.interpolate(
                cam, size=(h, w), mode="bilinear", align_corners=False
            )
            cam_np = _normalize_heatmap(cam_upsampled[0, 0].cpu().numpy())
    finally:
        h_fwd.remove()

    return cam_np, int(target_class_idx)


def explain_image(
    model: Any,
    input_batch: torch.Tensor,
    method: str = "saliency",
    target_class_idx: int | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Unified interface for generating explanations across all supported XAI methods."""
    method_key = method.lower().strip()
    if method_key not in AVAILABLE_METHODS:
        raise ValueError(
            f"Unknown XAI method '{method}'. Choose from: {AVAILABLE_METHODS}"
        )

    start_time = time.perf_counter()
    if method_key == "saliency":
        heatmap, class_idx = compute_saliency_map(
            model, input_batch, target_class_idx=target_class_idx
        )
    elif method_key == "gradcam":
        heatmap, class_idx = compute_gradcam(
            model, input_batch, target_class_idx=target_class_idx
        )
    elif method_key == "gradcam_pp":
        heatmap, class_idx = compute_gradcam_pp(
            model, input_batch, target_class_idx=target_class_idx
        )
    elif method_key == "scorecam":
        heatmap, class_idx = compute_scorecam(
            model,
            input_batch,
            target_class_idx=target_class_idx,
            batch_size=kwargs.get("batch_size", 64),
            max_channels=kwargs.get("max_channels", None),
        )
    else:
        raise ValueError(f"Unsupported method: {method_key}")
    elapsed = time.perf_counter() - start_time

    return {
        "method": method_key,
        "heatmap": heatmap,
        "target_class_idx": class_idx,
        "compute_time_sec": round(elapsed, 6),
    }


def plot_saliency_map(original_img, saliency_map, title, save_path):
    """Visualize the original image, pure heatmap, and the blended

    overlay side by side, perfectly aligned to original image dimensions.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_img)
    axes[0].axis("off")
    axes[0].set_title("Original Image")

    orig_width, orig_height = original_img.size

    norm_map = _normalize_heatmap(saliency_map)
    saliency_pil = Image.fromarray((255 * norm_map).astype(np.uint8))
    saliency_resized = saliency_pil.resize(
        (orig_width, orig_height), Image.Resampling.BILINEAR
    )
    saliency_resized_np = np.array(saliency_resized) / 255.0

    im1 = axes[1].imshow(saliency_resized_np, cmap="hot", vmin=0.0, vmax=1.0)
    axes[1].axis("off")
    axes[1].set_title("Pure Heatmap")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(original_img)
    im2 = axes[2].imshow(
        saliency_resized_np, cmap="hot", alpha=0.6, vmin=0.0, vmax=1.0
    )
    axes[2].axis("off")
    axes[2].set_title(title)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Visualization saved successfully to {save_path}")


def plot_methods_comparison(
    original_img: Image.Image,
    explanations: dict[str, dict[str, Any]],
    class_name: str,
    class_idx: int,
    save_path: str,
) -> None:
    """Plot original image alongside heatmaps/overlays for all evaluated XAI methods."""
    num_methods = len(explanations)
    fig, axes = plt.subplots(1, num_methods + 1, figsize=(4 * (num_methods + 1), 4.5))

    axes[0].imshow(original_img)
    axes[0].set_title(f"Original\n{class_name} ({class_idx})")
    axes[0].axis("off")

    orig_width, orig_height = original_img.size

    for ax, (method_key, info) in zip(axes[1:], explanations.items()):
        norm_map = _normalize_heatmap(info["heatmap"])
        pil_map = Image.fromarray((255 * norm_map).astype(np.uint8))
        resized = np.array(
            pil_map.resize((orig_width, orig_height), Image.Resampling.BILINEAR)
        ) / 255.0

        ax.imshow(original_img)
        ax.imshow(resized, cmap="jet", alpha=0.55, vmin=0.0, vmax=1.0)
        label = METHOD_LABELS.get(method_key, method_key)
        subtitle_parts = []
        if "deletion_auc" in info:
            subtitle_parts.append(f"Del:{info['deletion_auc']:.2f}")
        if "insertion_auc" in info:
            subtitle_parts.append(f"Ins:{info['insertion_auc']:.2f}")
        if "compute_time_sec" in info:
            subtitle_parts.append(f"{info['compute_time_sec']:.2f}s")
        subtitle = " | ".join(subtitle_parts)
        ax.set_title(f"{label}\n{subtitle}" if subtitle else label, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Comparison visualization saved to {save_path}")
