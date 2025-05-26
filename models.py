import torch
import torch.nn as nn
import open_clip
from pathlib import Path

# ------------------------------
# 1. MarineClassifier Model
# ------------------------------


class MarineClassifier(nn.Module):
    def __init__(self, unfreeze_top_x=None, taxonomy_tree=None):
        super(MarineClassifier, self).__init__()
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            "hf-hub:imageomics/bioclip"
        )

        # ✅ Freeze all layers initially
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # ✅ Dynamically unfreeze top X layers
        if unfreeze_top_x is not None:
            layers = list(self.clip_model.visual.transformer.resblocks.children())
            for layer in layers[-unfreeze_top_x:]:
                for param in layer.parameters():
                    param.requires_grad = True

        # ✅ Ensure taxonomy_tree is not None
        self.taxonomy_tree = taxonomy_tree if taxonomy_tree else {}

        # ✅ DEBUG: Print taxonomy tree structure before classifier initialization
        print(f"\n[DEBUG] Taxonomy Tree Loaded:\n{self.taxonomy_tree}")

        # ✅ Multi-rank classification head (handles missing ranks safely)
        self.fc = nn.ModuleDict(
            {
                rank: nn.Linear(
                    self.clip_model.visual.output_dim,
                    max(
                        2,
                        len(
                            set(
                                [
                                    taxonomy[rank]
                                    for taxonomy in self.taxonomy_tree.values()
                                    if rank in taxonomy
                                ]
                            )
                        ),
                    ),
                )
                for rank in [
                    "kingdom",
                    "phylum",
                    "class",
                    "order",
                    "family",
                    "genus",
                    "species",
                ]
            }
        )

        # ✅ DEBUG: Print initialized classifier heads and output neurons
        for rank, layer in self.fc.items():
            print(
                f"[DEBUG] Initialized Layer: '{rank}' → Output Classes: {layer.out_features}"
            )

    def forward(self, images):
        """Forward pass through BioCLIP & classifier head."""
        features = self.clip_model.encode_image(images)

        # ✅ DEBUG: Print shape of extracted features
        # print(f"[DEBUG] Extracted Features Shape: {features.shape}")

        # ✅ Ensure each rank always gets valid predictions
        outputs = {
            rank: (
                self.fc[rank](features)
                if rank in self.fc
                else torch.zeros(
                    features.shape[0],
                    max(1, len(set(self.taxonomy_tree.get(rank, {}).values()))),
                )
            )
            for rank in [
                "kingdom",
                "phylum",
                "class",
                "order",
                "family",
                "genus",
                "species",
            ]
        }

        # # ✅ DEBUG: Print model output keys
        # print(f"[DEBUG] Model Output: {outputs}")

        # # ✅ DEBUG: Verify logits shape for each rank
        # for rank, logits in outputs.items():
        #     print(f"[DEBUG] Rank '{rank}' → Logits: {logits.softmax(dim=1).cpu().tolist()}")
        # print("\n")
        return outputs


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
        print(f"model_path: {model_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Pretrained model '{model_name}.pth' not found")

        try:
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            print(f"Loaded weights from {model_path}")
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to load {model_path}. Check model arguments."
            ) from e

    # ✅ Print model size constraint but remove unnecessary check
    model_size_mb = calculate_model_size_mb(model)
    print(f"Model size: {model_size_mb:.2f} MB")

    return model
