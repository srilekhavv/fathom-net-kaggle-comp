import torch
import torch.nn as nn
import open_clip
from pathlib import Path


# ------------------------------
# 1. MarineClassifier Model
# ------------------------------


class MarineClassifier(nn.Module):
    def __init__(self, fine_tune_layers=None):
        """Initialize BioCLIP-based classifier.

        Args:
            num_classes (int): Number of output classes.
            fine_tune_layers (list, optional): Layers to unfreeze for fine-tuning.
        """
        super(MarineClassifier, self).__init__()
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            "hf-hub:imageomics/bioclip"
        )

        # Freeze BioCLIP layers initially
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Unfreeze selected layers for fine-tuning (if specified)
        if fine_tune_layers:
            for name, param in self.clip_model.named_parameters():
                if any(layer in name for layer in fine_tune_layers):
                    param.requires_grad = True

        # Classification head
        self.fc = nn.Linear(self.clip_model.visual.output_dim, 79)

    def forward(self, images):
        """Forward pass through BioCLIP & classifier head."""
        features = self.clip_model.encode_image(images)  # Extract embeddings
        return self.fc(features)  # Pass through classifier


# ------------------------------
# 2. Model Factory & Helpers
# ------------------------------

model_factory = {
    "classifier": MarineClassifier,
}


def calculate_model_size_mb(model: nn.Module) -> float:
    """Compute model size in MB."""
    return sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)


def save_model(model):
    """Save model checkpoint."""
    for name, model_class in model_factory.items():
        if isinstance(model, model_class):
            model_path = Path(__file__).resolve().parent / f"{name}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Model saved at {model_path}")
            return
    raise ValueError(f"Unsupported model type: {type(model)}")


def load_model(model_name: str, with_weights: bool = False, **model_kwargs):
    """Load a pre-trained model."""
    if model_name not in model_factory:
        raise ValueError(f"Model '{model_name}' not found in factory")

    model = model_factory[model_name](**model_kwargs)

    if with_weights:
        model_path = Path(__file__).resolve().parent / f"{model_name}.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Pretrained model '{model_name}.pth' not found")

        try:
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            print(f"Loaded weights from {model_path}")
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path}. Check default model arguments."
            ) from e

    # # Check model size constraint
    model_size_mb = calculate_model_size_mb(model)
    # if model_size_mb > 10:
    #     raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")
    print(f"Model size: {model_size_mb:.2f} MB")

    return model
