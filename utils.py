import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class LocalMarineDataset(Dataset):
    def __init__(
        self,
        base_path: str,
        annotations_filename: str,
        use_roi=False,
        transform=None,
        taxonomy_tree=None,
    ):
        """
        Args:
            base_path (str): Local dataset directory (e.g., '/content/dataset/train').
            annotations_filename (str): CSV file with image paths and labels.
            use_roi (bool): If True, load images from 'roi/' instead of 'images/'.
            transform (torchvision.transforms): Image preprocessing transforms.
            taxonomy_tree (dict): Reference taxonomy structure for hierarchical classification.
        """
        self.base_path = base_path
        self.use_roi = use_roi
        self.taxonomy_tree = taxonomy_tree

        # Load full annotations file
        annotations_path = os.path.join(base_path, annotations_filename)
        self.annotations = pd.read_csv(annotations_path)

        # ✅ Get the list of actually downloaded images
        folder = "rois" if self.use_roi else "images"
        downloaded_images = set(os.listdir(os.path.join(base_path, folder)))

        # ✅ Filter annotations to only include existing images
        self.annotations = self.annotations[
            self.annotations["path"].apply(
                lambda x: os.path.basename(x) in downloaded_images
            )
        ]

        # ✅ Store hierarchical labels
        self.label_mapping = {
            rank: {label: idx for idx, label in enumerate(sorted(classes))}
            for rank, classes in taxonomy_tree.items()
        }

        # Define transformation
        self.transform = (
            transform
            if transform
            else transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        )

    def __len__(self):
        """Return the number of available images after filtering."""
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]

        # ✅ Ensure label exists in ANY taxonomic rank
        label_found = any(
            row["label"] in rank_labels for rank_labels in self.taxonomy_tree.values()
        )
        if pd.isna(row["label"]) or not label_found:
            print(
                f"Warning: Unrecognized label '{row['label']}' at index {idx}, skipping."
            )
            return None  # ✅ Skip bad samples

        # Load image
        folder = "rois" if self.use_roi else "images"
        filename = os.path.basename(row["path"])
        image_path = os.path.join(self.base_path, folder, filename)

        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: Image '{image_path}' not found, skipping.")
            return None  # ✅ Skip missing files safely

        image = self.transform(image)

        # ✅ Encode hierarchical labels
        labels = {
            rank: torch.tensor(self.label_mapping[rank][row["label"]], dtype=torch.long)
            for rank in self.taxonomy_tree.keys()
            if row["label"] in self.label_mapping[rank]
        }

        return image, labels


# ------------------------------
# 5. Dataloader Function
# ------------------------------


def load_data(
    base_path: str,
    annotations_filename: str,
    batch_size: int = 128,
    shuffle: bool = False,
    use_roi: bool = True,
    taxonomy_tree=None,
) -> DataLoader:
    """Create a DataLoader for training with hierarchical taxonomy."""
    dataset = LocalMarineDataset(
        base_path, annotations_filename, use_roi, taxonomy_tree=taxonomy_tree
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def create_taxonomic_mapping(labels):
    """Sorts labels into their correct taxonomic rank."""
    taxonomy_tree = {
        "kingdom": set(),
        "phylum": set(),
        "class": set(),
        "order": set(),
        "family": set(),
        "genus": set(),
        "species": set(),
    }

    for label in labels:
        rank = classify_taxonomic_rank(label)
        taxonomy_tree[rank].add(label)

    return taxonomy_tree


def classify_taxonomic_rank(label):
    """Classifies a label into one of the taxonomic ranks."""
    if " " in label:
        return "species"
    elif label.endswith("idae"):
        return "family"
    elif label.endswith("formes"):
        return "order"
    elif label.endswith("phyta") or label.endswith("mycota"):
        return "phylum"
    elif label.endswith("aceae"):
        return "class"
    else:
        return "genus"  # Default assumption
