import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from fathomnet.api import worms


def extract_taxonomic_ranks(node, target_ranks, taxonomy_dict):
    """Recursively traverse WoRMSNode hierarchy and extract only target ranks."""
    if node.rank.lower() in target_ranks:
        taxonomy_dict[node.rank.lower()] = node.name

    for child in node.children:
        extract_taxonomic_ranks(child, target_ranks, taxonomy_dict)


def get_taxonomic_tree(labels):
    """
    Fetch taxonomic hierarchy using WoRMS `get_ancestors()` and extract only:
    ["kingdom", "phylum", "class", "order", "family", "genus", "species"].
    """
    target_ranks = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
    taxonomy_tree = {}

    for label in labels:
        raw_ancestors = worms.get_ancestors(label)
        taxonomy_dict = {}  # ✅ Store full hierarchy for each label
        extract_taxonomic_ranks(raw_ancestors, target_ranks, taxonomy_dict)

        taxonomy_tree[label] = taxonomy_dict  # ✅ Save full ancestry per label

    print(f"\n[DEBUG] Taxonomy Tree Extracted:\n{taxonomy_tree}\n")  # ✅ Debug print
    return taxonomy_tree


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

        # ✅ Load full annotations file
        annotations_path = os.path.join(base_path, annotations_filename)
        self.annotations = pd.read_csv(annotations_path)

        # ✅ Get the list of actually downloaded images
        folder = "rois" if self.use_roi else "images"
        downloaded_images = set(os.listdir(os.path.join(base_path, folder)))

        # ✅ Filter annotations to include only existing images
        self.annotations = self.annotations[
            self.annotations["path"].apply(
                lambda x: os.path.basename(x) in downloaded_images
            )
        ]

        # ✅ Fix paths for Google Colab
        self.annotations["path"] = self.annotations["path"].apply(
            lambda x: f"/content/fathom-net-kaggle-comp/dataset/train/{folder}/{os.path.basename(x)}"
        )

        # ✅ Fetch taxonomic hierarchy dynamically
        if taxonomy_tree is None:
            self.taxonomy_tree = get_taxonomic_tree(self.annotations["label"].unique())

        print(
            f"\n[DEBUG] Taxonomy Tree Loaded in Dataset:\n{self.taxonomy_tree}\n"
        )  # ✅ Debug print

        # ✅ Encode labels based on full ancestry
        self.label_mapping = {
            rank: {
                label: idx
                for idx, label in enumerate(
                    sorted(
                        set(
                            self.taxonomy_tree[label][rank]
                            for label in self.taxonomy_tree
                            if rank in self.taxonomy_tree[label]
                        )
                    )
                )
            }
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

        print("\n[DEBUG] Label Mapping:")
        for rank, mapping in self.label_mapping.items():
            print(f"{rank}: {mapping}")  # ✅ Debug print

        # ✅ Define transformation
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

        # ✅ Double-check if the file still exists
        image_path = row["path"]
        if not os.path.exists(image_path):
            print(f"Warning: Image '{image_path}' not found, skipping.")
            return None  # ✅ Skip missing files safely

        image = Image.open(image_path).convert("RGB")
        image = (
            self.transform(image) if self.transform else transforms.ToTensor()(image)
        )

        labels = {}
        for rank in [
            "kingdom",
            "phylum",
            "class",
            "order",
            "family",
            "genus",
            "species",
        ]:
            if rank in self.taxonomy_tree[row["label"]]:  # ✅ Lookup ancestry correctly
                labels[rank] = torch.tensor(
                    self.label_mapping[rank][self.taxonomy_tree[row["label"]][rank]],
                    dtype=torch.long,
                )
            else:
                unknown_label = f"Unknown_{rank}"
                labels[rank] = torch.tensor(
                    self.label_mapping[rank].get(
                        unknown_label, len(self.label_mapping[rank])
                    ),
                    dtype=torch.long,
                )

        print(
            f"[DEBUG] Labels for idx={idx}: {labels}"
        )  # ✅ Debug print before returning labels
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
        base_path, annotations_filename, use_roi, taxonomy_tree
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
