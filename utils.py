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
        """Initialize dataset with taxonomy-aware hierarchical labels."""
        self.base_path = base_path
        self.use_roi = use_roi
        self.taxonomy_tree = taxonomy_tree

        # ✅ Load and filter annotations
        annotations_path = os.path.join(base_path, annotations_filename)
        self.annotations = pd.read_csv(annotations_path)

        folder = "rois" if self.use_roi else "images"
        downloaded_images = set(os.listdir(os.path.join(base_path, folder)))
        self.annotations = self.annotations[
            self.annotations["path"].apply(
                lambda x: os.path.basename(x) in downloaded_images
            )
        ]

        self.annotations["path"] = self.annotations["path"].apply(
            lambda x: f"{base_path}/{folder}/{os.path.basename(x)}"
        )

        # ✅ Fetch taxonomic hierarchy dynamically if not provided
        if taxonomy_tree is None:
            self.taxonomy_tree = get_taxonomic_tree(self.annotations["label"].unique())

        missing_labels = set(self.annotations["label"].unique()) - set(
            self.taxonomy_tree.keys()
        )
        print(f"[DEBUG] Missing Species Labels in taxonomy_tree: {missing_labels}")

        # ✅ Collect all unique labels across taxonomy ranks
        all_labels = {
            rank: set()
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

        for label, taxonomy in self.taxonomy_tree.items():
            for rank in all_labels:
                if rank in taxonomy:
                    all_labels[rank].add(taxonomy[rank])

        # ✅ Ensure label mapping contains all known labels before training
        self.label_mapping = {
            rank: {label: idx for idx, label in enumerate(sorted(all_labels[rank]))}
            for rank in all_labels
        }

        for rank in [
            "kingdom",
            "phylum",
            "class",
            "order",
            "family",
            "genus",
            "species",
        ]:
            extracted_labels = set(
                self.taxonomy_tree[label][rank]
                for label in self.taxonomy_tree
                if rank in self.taxonomy_tree[label]
            )

            if not extracted_labels.issubset(set(self.label_mapping[rank].keys())):
                print(
                    f"[ERROR] Rank '{rank}' has taxonomy labels missing from label_mapping!"
                )
        print(f"[DEBUG] label_mapping: {self.label_mapping}")
        # ✅ Define image transformation pipeline
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        """Return number of available images."""
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]

        # ✅ Verify image existence
        image_path = row["path"]
        if not os.path.exists(image_path):
            return None  # Skip missing images safely

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # ✅ Retrieve taxonomy labels per rank safely
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
            true_label = self.taxonomy_tree[row["label"]].get(rank, None)

            if true_label is None:
                # print(f"[WARNING] Rank '{rank}' missing for label '{row['label']}'")
                labels[rank] = torch.tensor(
                    -1, dtype=torch.long
                )  # ✅ Assign default safe value (-1)
            elif true_label not in self.label_mapping[rank]:
                # print(f"[WARNING] Rank '{rank}' → Label '{true_label}' not found in label_mapping!")
                labels[rank] = torch.tensor(
                    -1, dtype=torch.long
                )  # ✅ Assign default safe value (-1)
            else:
                labels[rank] = torch.tensor(
                    self.label_mapping[rank][true_label], dtype=torch.long
                )

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
