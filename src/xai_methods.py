from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch


def compute_saliency_map(model, input_batch, target_class_idx=None):
    model.eval()
    input_tensor = input_batch.clone().detach()
    input_tensor.requires_grad_()
    output = model(input_tensor)

    if target_class_idx is None:
        target_class_idx = output.argmax(dim=1).item()

    score = output[0, target_class_idx]
    model.zero_grad()
    if input_tensor.grad is not None:
        input_tensor.grad.zero_()

    score.backward()

    assert input_tensor.grad is not None
    gradients = input_tensor.grad.data.abs()
    saliency, _ = torch.max(gradients, dim=1)
    saliency_np = saliency[0].cpu().numpy()

    return saliency_np, target_class_idx


def plot_saliency_map(original_img, saliency_map, title, save_path):
    """Visualize the original image, pure saliency heatmap, and the blended

    overlay side by side, perfectly aligned to original image dimensions.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_img)
    axes[0].axis("off")
    axes[0].set_title("Original Image")

    orig_width, orig_height = original_img.size

    saliency_pil = Image.fromarray((255 * saliency_map).astype(np.uint8))
    saliency_resized = saliency_pil.resize(
        (orig_width, orig_height), Image.Resampling.BILINEAR
    )
    saliency_resized_np = np.array(saliency_resized) / 255.0

    im1 = axes[1].imshow(saliency_resized_np, cmap="hot")
    axes[1].axis("off")
    axes[1].set_title("Pure Heatmap")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(original_img)
    im2 = axes[2].imshow(saliency_resized_np, cmap="hot", alpha=0.6)
    axes[2].axis("off")
    axes[2].set_title(title)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Visualization saved successfully to {save_path}")
