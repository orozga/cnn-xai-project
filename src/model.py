import torch
from torchvision.models import ResNet50_Weights, resnet50


class XAIResNetModel:
    """Manage a ResNet-50 model and hooks for XAI feature extraction."""

    def __init__(self, device: str | None = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=weights)
        self.model.to(self.device)
        self.model.eval()

        self.feature_maps = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        """Register hooks on the final convolutional block."""

        def forward_hook(module, input, output):
            self.feature_maps = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        target_layer = self.model.layer4
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def predict(self, input_tensor: torch.Tensor):
        """Run classification for an image tensor."""
        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
        return output


if __name__ == "__main__":
    print("Initializing ResNet-50...")
    xai_model = XAIResNetModel()
    print(f"Model loaded successfully on device: {xai_model.device}")

    dummy_input = torch.randn(1, 3, 224, 224)
    output = xai_model.predict(dummy_input)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    _, top_categories = torch.topk(probabilities, 5)

    print("\nForward pass completed successfully!")
    print(f"Prediction output shape: {output.shape}")
    print(f"Top category indices: {top_categories.tolist()}")
    if xai_model.feature_maps is not None:
        print(f"Captured feature map shape: {xai_model.feature_maps.shape}")
