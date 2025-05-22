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
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Set random seed for deterministic training
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Directory for logging (with timestamp)
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # Load model and move to device
    model = load_model(model_name, **kwargs).to(device)
    # for name, param in model.clip_model.named_parameters():
    #     print(name)
    # return
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
    metrics = {
        "train_acc": [],
        "train_loss": [],
        "train_precision": [],
        "train_recall": [],
        "train_f1": [],
        "val_acc": [],
        "val_loss": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
    }

    print("Starting training loop")
    # Training loop
    for epoch in range(num_epoch):
        for key in metrics:
            metrics[key].clear()  # Clear metrics at start of each epoch

        model.train()
        all_train_labels, all_train_preds = [], []

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
            train_acc = outputs.argmax(dim=1).eq(label).float().mean().item()
            metrics["train_acc"].append(train_acc)
            metrics["train_loss"].append(loss_val.item())

            # Store predictions for precision/recall calculations
            all_train_labels.extend(label.cpu().numpy())
            all_train_preds.extend(outputs.argmax(dim=1).cpu().numpy())

            global_step += 1

        # Compute precision, recall, F1-score for training
        train_precision = precision_score(
            all_train_labels, all_train_preds, average="macro", zero_division=0
        )
        train_recall = recall_score(
            all_train_labels, all_train_preds, average="macro", zero_division=0
        )
        train_f1 = f1_score(
            all_train_labels, all_train_preds, average="macro", zero_division=0
        )

        metrics["train_precision"].append(train_precision)
        metrics["train_recall"].append(train_recall)
        metrics["train_f1"].append(train_f1)

        # Evaluate model on validation set
        with torch.inference_mode():
            model.eval()
            all_val_labels, all_val_preds = [], []
            for img, label in val_dataloader:
                img, label = img.to(device), label.to(device)

                # Compute validation accuracy
                pred = model(img)
                val_acc = pred.argmax(dim=1).eq(label).float().mean().item()
                val_loss = loss_func(pred, label).item()

                metrics["val_acc"].append(val_acc)
                metrics["val_loss"].append(val_loss)

                all_val_labels.extend(label.cpu().numpy())
                all_val_preds.extend(pred.argmax(dim=1).cpu().numpy())

            # Compute precision, recall, F1-score for validation
            val_precision = precision_score(
                all_val_labels, all_val_preds, average="macro", zero_division=0
            )
            val_recall = recall_score(
                all_val_labels, all_val_preds, average="macro", zero_division=0
            )
            val_f1 = f1_score(
                all_val_labels, all_val_preds, average="macro", zero_division=0
            )

            metrics["val_precision"].append(val_precision)
            metrics["val_recall"].append(val_recall)
            metrics["val_f1"].append(val_f1)

        # Log metrics to TensorBoard
        logger.add_scalar("train/accuracy", np.mean(metrics["train_acc"]), global_step)
        logger.add_scalar("train/loss", np.mean(metrics["train_loss"]), global_step)
        logger.add_scalar("train/precision", train_precision, global_step)
        logger.add_scalar("train/recall", train_recall, global_step)
        logger.add_scalar("train/f1-score", train_f1, global_step)

        logger.add_scalar("val/accuracy", np.mean(metrics["val_acc"]), global_step)
        logger.add_scalar("val/loss", np.mean(metrics["val_loss"]), global_step)
        logger.add_scalar("val/precision", val_precision, global_step)
        logger.add_scalar("val/recall", val_recall, global_step)
        logger.add_scalar("val/f1-score", val_f1, global_step)

        # Print progress
        print(
            f"Epoch {epoch + 1}/{num_epoch}: train_acc={np.mean(metrics['train_acc']):.4f}, val_acc={np.mean(metrics['val_acc']):.4f}, train_f1={train_f1:.4f}, val_f1={val_f1:.4f}"
        )

    # Save model checkpoints
    save_model(model)
    torch.save(model.state_dict(), log_dir / f"{model_name}.pth")
    print(f"Model saved to {log_dir / f'{model_name}.pth'}")
