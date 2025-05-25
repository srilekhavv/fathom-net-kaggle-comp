import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from datetime import datetime
from pathlib import Path
import torch.utils.tensorboard as tb
from sklearn.metrics import precision_score, recall_score, f1_score

from models import load_model, save_model
from utils import load_data
from metrics import hierarchical_loss, compute_accuracy
from torch.utils.data import random_split


def train(
    exp_dir: str = "logs",
    model_name: str = "classifier",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    dataset_path: str = "/content/fathom-net-kaggle-comp/dataset/",
    taxonomy_tree=None,
    **kwargs,
):
    """Train the model with hierarchical accuracy and structured logging."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ✅ Create logging directory
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # ✅ Load model
    model = load_model(model_name, **kwargs).to(device)
    model.train()

    # ✅ Load dataset
    full_train_dataset = load_data(
        os.path.join(dataset_path, "train"),
        "annotations.csv",
        batch_size=batch_size,
        use_roi=True,
    )

    # ✅ Split into train & validation sets (80/20)
    train_size = int(0.8 * len(full_train_dataset.dataset))
    val_size = len(full_train_dataset.dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset.dataset, [train_size, val_size]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    # ✅ Loss function & optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0

    print("Starting training loop")
    for epoch in range(num_epoch):
        model.train()
        total_train_loss, total_val_loss = 0, 0
        total_train_acc, total_val_acc = {}, {}

        # ✅ Initialize accuracy tracking per rank
        for rank in [
            "kingdom",
            "phylum",
            "class",
            "order",
            "family",
            "genus",
            "species",
        ]:
            total_train_acc[rank] = 0
            total_val_acc[rank] = 0

        num_train_batches, num_val_batches = 0, 0

        # ✅ Training loop
        for img, ground_truth in train_loader:
            img = img.to(device)
            predictions = model(img)

            loss, _ = hierarchical_loss(predictions, ground_truth, taxonomy_tree)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # ✅ Compute accuracy per taxonomy rank
            accuracy = compute_accuracy(predictions, ground_truth)
            for rank in total_train_acc:
                if accuracy[rank] is not None:
                    total_train_acc[rank] += accuracy[rank]

            total_train_loss += loss.item()
            num_train_batches += 1
            global_step += 1

        # ✅ Normalize accuracy values
        avg_train_acc = {
            rank: (
                (total_train_acc[rank] / num_train_batches) * 100
                if num_train_batches > 0
                else 0
            )
            for rank in total_train_acc
        }

        # ✅ Log training metrics
        logger.add_scalar(
            "train/loss", total_train_loss / num_train_batches, global_step
        )
        for rank in avg_train_acc:
            logger.add_scalar(
                f"train/accuracy_{rank}", avg_train_acc[rank], global_step
            )

        print(
            f"Epoch {epoch+1}/{num_epoch}: train_loss={total_train_loss / num_train_batches:.4f}, train_accuracy={avg_train_acc}"
        )

        # ✅ Validation loop
        model.eval()
        with torch.no_grad():
            for img, ground_truth in val_loader:
                img = img.to(device)
                predictions = model(img)

                val_loss, _ = hierarchical_loss(
                    predictions, ground_truth, taxonomy_tree
                )

                # ✅ Compute accuracy per taxonomy rank
                val_accuracy = compute_accuracy(predictions, ground_truth)
                for rank in total_val_acc:
                    if val_accuracy[rank] is not None:
                        total_val_acc[rank] += val_accuracy[rank]

                total_val_loss += val_loss.item()
                num_val_batches += 1

        avg_val_acc = {
            rank: (
                (total_val_acc[rank] / num_val_batches) * 100
                if num_val_batches > 0
                else 0
            )
            for rank in total_val_acc
        }

        # ✅ Log validation metrics
        logger.add_scalar("val/loss", total_val_loss / num_val_batches, global_step)
        for rank in avg_val_acc:
            logger.add_scalar(f"val/accuracy_{rank}", avg_val_acc[rank], global_step)

        print(
            f"Epoch {epoch+1}/{num_epoch}: val_loss={total_val_loss / num_val_batches:.4f}, val_accuracy={avg_val_acc}"
        )

        # ✅ Save model checkpoint
        model_path = log_dir / f"{model_name}_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), model_path)
        save_model(model)
        print(f"Model checkpoint saved: {model_path}")
