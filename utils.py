import os
import io
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from google.cloud import storage
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ------------------------------
# 1. Google Cloud Authentication
# ------------------------------


def authenticate_gcs(service_account_json=None):
    """Authenticate with Google Cloud using a service account key."""
    if service_account_json:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_json
    try:
        storage_client = storage.Client()
        _ = storage_client.list_buckets()
        print("Google Cloud Authentication Successful!")
    except Exception as e:
        print("Google Cloud Authentication Failed:", e)
        print("Please provide a valid service account key.")


# Call authentication once when utils is imported
SERVICE_ACCOUNT_PATH = "fathom-net-kaggle-9a123ad1b993.json"  # Update with correct path
authenticate_gcs(SERVICE_ACCOUNT_PATH)

# ------------------------------
# 2. Google Cloud Storage Helpers
# ------------------------------


def get_gcs_client():
    """Returns an authenticated Google Cloud Storage client."""
    return storage.Client()


def load_gcs_csv(bucket_name: str, file_path: str) -> pd.DataFrame:
    """Fetch CSV file from GCS and load it into a Pandas DataFrame."""
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    csv_bytes = blob.download_as_bytes()
    return pd.read_csv(io.BytesIO(csv_bytes))


def load_gcs_image(bucket_name: str, file_path: str) -> Image.Image:
    """Fetch image from GCS and return a PIL Image."""
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    image_bytes = blob.download_as_bytes()
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


# ------------------------------
# 3. Data Processing & Encoding
# ------------------------------


def create_label_mapping(annotations: pd.DataFrame) -> dict:
    """Generate a mapping from class labels to numerical indices."""
    return {name: idx for idx, name in enumerate(annotations["label"].unique())}


def encode_label(label: str, label_mapping: dict) -> torch.Tensor:
    """Convert a class label into a numerical index tensor."""
    return torch.tensor(label_mapping[label], dtype=torch.long)


# ------------------------------
# 4. Dataset Class for GCS
# ------------------------------


class GCSMarineDataset(Dataset):
    def __init__(
        self,
        bucket_name: str,
        data_folder: str,
        annotations_filename: str,
        transform=None,
    ):
        self.bucket_name = bucket_name
        self.data_folder = data_folder
        self.annotations = load_gcs_csv(
            bucket_name, f"{data_folder}/{annotations_filename}"
        )
        self.transform = transform if transform else transforms.ToTensor()
        self.label_mapping = create_label_mapping(self.annotations)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        image_path = f"{self.data_folder}/images/{row['path']}"  # Image path in GCS
        image = load_gcs_image(self.bucket_name, image_path)  # Fetch image
        image = self.transform(image)
        label = encode_label(row["label"], self.label_mapping)  # Encode label
        return image, label


# ------------------------------
# 5. Dataloader Function
# ------------------------------


def load_data(
    bucket_name: str,
    data_folder: str,
    annotations_filename: str,
    batch_size: int = 128,
    shuffle: bool = False,
) -> DataLoader:
    """Create a DataLoader for training with GCS-stored images."""
    dataset = GCSMarineDataset(bucket_name, data_folder, annotations_filename)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


# ------------------------------
# 6. Evaluation Metrics
# ------------------------------


def compute_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute classification accuracy from model predictions.
    Arguments:
        outputs: torch.Tensor of logits/probabilities (shape: batch_size, num_classes)
        labels: torch.Tensor of true class labels (shape: batch_size,)
    Returns:
        Accuracy as a scalar tensor.
    """
    outputs_idx = outputs.argmax(dim=1).type_as(labels)
    return (outputs_idx == labels).float().mean()
