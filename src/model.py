import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class XAIResNetModel:
    """Manage a ResNet-50 model and hooks for XAI feature and gradient extraction."""

    def __init__(
        self,
        device: str | None = None,
        target_layer_name: str = "layer4",
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=weights)
        self.model.to(self.device)
        self.model.eval()

        self.feature_maps: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._forward_handle = None
        self._backward_handle = None
        self.target_layer_name = target_layer_name
        self._register_hooks(target_layer_name)

    def _resolve_layer(self, layer_name: str) -> nn.Module:
        modules = dict(self.model.named_modules())
        if layer_name not in modules:
            raise ValueError(
                f"Layer '{layer_name}' not found in ResNet-50. "
                f"Available top-level blocks: layer1, layer2, layer3, layer4."
            )
        return modules[layer_name]

    def _register_hooks(self, layer_name: str = "layer4") -> None:
        """Register forward and backward hooks on the selected convolutional block."""
        self.remove_hooks()

        def forward_hook(module, input, output):
            self.feature_maps = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        target_layer = self._resolve_layer(layer_name)
        self._forward_handle = target_layer.register_forward_hook(forward_hook)
        self._backward_handle = target_layer.register_full_backward_hook(backward_hook)
        self.target_layer_name = layer_name

    def set_target_layer(self, layer_name: str) -> None:
        """Switch the monitored convolutional layer dynamically."""
        self.feature_maps = None
        self.gradients = None
        self._register_hooks(layer_name)

    def remove_hooks(self) -> None:
        """Remove registered forward and backward hooks."""
        if self._forward_handle is not None:
            self._forward_handle.remove()
            self._forward_handle = None
        if self._backward_handle is not None:
            self._backward_handle.remove()
            self._backward_handle = None

    def predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run classification for an image tensor without tracking gradients."""
        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
        return output

    def forward_and_backward(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: int | None = None,
    ) -> tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
        """Run forward and backward passes and return logits, class_idx, activations, and gradients."""
        self.model.eval()
        self.feature_maps = None
        self.gradients = None

        x = input_tensor.to(self.device)
        output = self.model(x)

        if target_class_idx is None:
            target_class_idx = int(output.argmax(dim=1).item())

        self.model.zero_grad()
        score = output[0, target_class_idx]
        score.backward()

        if self.feature_maps is None or self.gradients is None:
            raise RuntimeError(
                f"Hooks on layer '{self.target_layer_name}' did not capture activations or gradients."
            )

        return (
            output.detach(),
            int(target_class_idx),
            self.feature_maps.detach(),
            self.gradients.detach(),
        )


if __name__ == "__main__":
    print("Initializing ResNet-50...")
    xai_model = XAIResNetModel()
    print(f"Model loaded successfully on device: {xai_model.device}")
    print(f"Hooked target layer: {xai_model.target_layer_name}")

    dummy_input = torch.randn(1, 3, 224, 224)
    logits, top_class, fmap, grads = xai_model.forward_and_backward(dummy_input)
    probabilities = torch.nn.functional.softmax(logits[0], dim=0)
    _, top_categories = torch.topk(probabilities, 5)

    print("\nForward & backward passes completed successfully!")
    print(f"Prediction output shape: {logits.shape}")
    print(f"Top category indices: {top_categories.tolist()} (target: {top_class})")
    print(f"Captured feature map shape: {fmap.shape}")
    print(f"Captured gradients shape: {grads.shape}")
