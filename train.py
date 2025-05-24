import os
import torch
import numpy as np
from pathlib import Path
import torch.utils.tensorboard as tb
from torch.utils.data import random_split

from models import load_model
from utils import load_data, get_taxonomic_tree
from metrics import hierarchical_loss


def train(
    exp_dir="logs",
    model_name="classifier",
    num_epoch=50,
    lr=1e-3,
    batch_size=128,
    seed=2024,
    dataset_path="/content/fathom-net-kaggle-comp/dataset/",
    **model_kwargs,
):
    """Train the model using hierarchical classification."""

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ✅ Load dataset first
    full_train_dataset = load_data(
        os.path.join(dataset_path, "train"),
        "annotations.csv",
        batch_size=batch_size,
        use_roi=True,
    )

    # ✅ Generate taxonomy tree **only once**
    taxonomy_tree = get_taxonomic_tree(
        full_train_dataset.dataset.annotations["label"].unique()
    )

    # ✅ Pass the same taxonomy tree to the dataset to avoid duplicate calls
    full_train_dataset.dataset.taxonomy_tree = taxonomy_tree

    model = load_model("classifier", taxonomy_tree=taxonomy_tree, **model_kwargs).to(
        device
    )
    model.train()

    # Split train & validation datasets
    train_size = int(0.8 * len(full_train_dataset.dataset))
    val_size = len(full_train_dataset.dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset.dataset, [train_size, val_size]
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Starting hierarchical training loop")
    for epoch in range(num_epoch):
        model.train()
        for img, labels in train_dataloader:
            img, labels = img.to(device), {
                rank: labels[rank].to(device) for rank in labels.keys()
            }

            # Forward pass
            outputs = model(img)
            loss_val, distance_val = hierarchical_loss(outputs, labels, taxonomy_tree)

            # Backward pass
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

        # Evaluate model
        with torch.inference_mode():
            model.eval()
            val_losses = []
            val_distances = []
            for img, labels in val_dataloader:
                img, labels = img.to(device), {
                    rank: labels[rank].to(device) for rank in labels.keys()
                }
                outputs = model(img)
                val_loss, val_distance = hierarchical_loss(
                    outputs, labels, taxonomy_tree
                )
                val_losses.append(val_loss.item())
                val_distances.append(val_distance)

        print(
            f"Epoch {epoch+1}/{num_epoch}: train_loss={loss_val:.4f}, val_loss={np.mean(val_losses):.4f}, avg_taxonomic_distance={np.mean(val_distances):.4f}"
        )

    # Save model
    torch.save(model.state_dict(), Path(exp_dir) / f"{model_name}.pth")
