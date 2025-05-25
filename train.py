import os
import torch
import numpy as np
from pathlib import Path
import torch.utils.tensorboard as tb
from torch.utils.data import random_split

from models import load_model
from utils import load_data, get_taxonomic_tree
from metrics import hierarchical_loss, compute_accuracy
from datetime import datetime


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

    # ✅ Pass the same taxonomy tree to the dataset to avoid duplicate calls
    taxonomy_tree = full_train_dataset.dataset.taxonomy_tree

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

    # ✅ Create TensorBoard Logger
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    print("Starting hierarchical training loop")
    for epoch in range(num_epoch):
        model.train()
        total_train_acc = {
            rank: 0
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
        num_train_batches = 0

        # ✅ Training loop
        for img, labels in train_dataloader:

            img, labels = img.to(device), {
                rank: labels[rank].to(device) for rank in labels.keys()
            }
            # print(f"[DEBUG] TRAINING: {labels}")
            # Forward pass
            outputs = model(img)
            loss_val, distance_val = hierarchical_loss(outputs, labels, taxonomy_tree)

            # Backward pass
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # ✅ Compute accuracy per taxonomy rank
            accuracy = compute_accuracy(outputs, labels)
            for rank in total_train_acc:
                if accuracy[rank] is not None:
                    total_train_acc[rank] += accuracy[rank]

            num_train_batches += 1

        # ✅ Normalize accuracy values
        avg_train_acc = {
            rank: (total_train_acc[rank] / num_train_batches) * 100
            for rank in total_train_acc
        }

        # ✅ Log training metrics
        logger.add_scalar("train/loss", loss_val.item(), epoch)
        for rank in avg_train_acc:
            logger.add_scalar(f"train/accuracy_{rank}", avg_train_acc[rank], epoch)

        # ✅ Validation loop in inference mode
        model.eval()
        val_losses, val_distances = [], []
        total_val_acc = {
            rank: 0
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
        num_val_batches = 0

        with torch.inference_mode():
            for img, labels in val_dataloader:
                # print(f"[DEBUG] VAL")
                img, labels = img.to(device), {
                    rank: labels[rank].to(device) for rank in labels.keys()
                }
                outputs = model(img)
                val_loss, val_distance = hierarchical_loss(
                    outputs, labels, taxonomy_tree
                )

                # ✅ Compute accuracy per taxonomy rank
                val_accuracy = compute_accuracy(outputs, labels)
                for rank in total_val_acc:
                    if val_accuracy[rank] is not None:
                        total_val_acc[rank] += val_accuracy[rank]

                val_losses.append(val_loss.item())
                val_distances.append(val_distance)
                num_val_batches += 1

        # ✅ Normalize accuracy values
        avg_val_acc = {
            rank: (total_val_acc[rank] / num_val_batches) * 100
            for rank in total_val_acc
        }

        # ✅ Log validation metrics
        logger.add_scalar("val/loss", np.mean(val_losses), epoch)
        for rank in avg_val_acc:
            logger.add_scalar(f"val/accuracy_{rank}", avg_val_acc[rank], epoch)

        print(
            f"Epoch {epoch+1}/{num_epoch}: "
            f"train_loss={loss_val:.4f}, val_loss={np.mean(val_losses):.4f}, "
            f"avg_taxonomic_distance={np.mean(val_distances):.4f}, "
            f"train_acc={avg_train_acc}, val_acc={avg_val_acc}"
        )

    # Save model
    torch.save(model.state_dict(), log_dir / f"{model_name}.pth")
    print(f"Model saved to {log_dir / f'{model_name}.pth'}")
