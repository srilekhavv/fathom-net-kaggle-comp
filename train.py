import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from datetime import datetime
from pathlib import Path
import torch.utils.tensorboard as tb

from models import load_model, save_model
from utils import load_data, compute_accuracy
from torch.utils.data import random_split


def train(
    exp_dir: str = "logs",
    model_name: str = "classifier",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    dataset_path: str = "/content/fathom-net-kaggle-comp/dataset/",
    **kwargs,
):
    """Train the model using a structured Train/Validation/Test split."""

    # Select device (GPU or CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # Set random seed for deterministic training
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Directory for logging (with timestamp)
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # Load model and move to device
    model = load_model(model_name, **kwargs).to(device)
    model.train()

    # Load full training dataset
    full_train_dataset = load_data(
        os.path.join(dataset_path, "train"),
        "annotations.csv",
        batch_size=batch_size,
        use_roi=True,
    )

    # **Split full train set into train & validation (80/20)**
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

    # Loss function & optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    metrics = {"train_acc": [], "val_acc": []}

    print("Starting training loop")
    # Training loop
    for epoch in range(num_epoch):
        for key in metrics:
            metrics[key].clear()  # Clear metrics at start of each epoch

        model.train()
        for img, label in train_dataloader:
            img, label = img.to(device), label.to(device)

            # Forward pass
            outputs = model(img)
            loss_val = loss_func(outputs, label)

            # Backward pass
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Compute training accuracy
            train_acc = (
                (outputs.argmax(dim=1).type_as(label) == label).float().mean().item()
            )
            metrics["train_acc"].append(train_acc)

            global_step += 1

        # Evaluate model on validation set
        with torch.inference_mode():
            model.eval()
            for img, label in val_dataloader:
                img, label = img.to(device), label.to(device)

                # Compute validation accuracy
                pred = model(img)
                val_acc = (
                    (pred.argmax(dim=1).type_as(label) == label).float().mean().item()
                )
                metrics["val_acc"].append(val_acc)

        # Log metrics to TensorBoard
        epoch_train_acc = torch.tensor(metrics["train_acc"]).mean()
        epoch_val_acc = torch.tensor(metrics["val_acc"]).mean()

        logger.add_scalar("train_acc", epoch_train_acc.item(), global_step)
        logger.add_scalar("val_acc", epoch_val_acc.item(), global_step)

        # Print progress
        print(
            f"Epoch {epoch + 1}/{num_epoch}: train_acc={epoch_train_acc:.4f}, val_acc={epoch_val_acc:.4f}"
        )

    # Save model checkpoints
    save_model(model)
    torch.save(model.state_dict(), log_dir / f"{model_name}.pth")
    print(f"Model saved to {log_dir / f'{model_name}.pth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--dataset_path", type=str, default="/content/fathom-net-kaggle-comp/dataset/"
    )

    # Parse arguments and run training
    train(**vars(parser.parse_args()))
