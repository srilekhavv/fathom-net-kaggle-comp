import torch.nn.functional as F


import torch.nn.functional as F


def hierarchical_loss(predictions, ground_truth, taxonomy_tree, distance_penalty=0.1):
    """Computes taxonomic-aware loss based on hierarchical distance, safely handling missing labels (-1)."""
    total_loss = 0
    total_distance = 0

    # for rank, logits in predictions.items():
    #     pred_class = logits.argmax(dim=1).cpu().tolist()
    #     true_class = ground_truth[rank].cpu().tolist()
    #     print(f"[DEBUG] Rank '{rank}' → Predicted: {pred_class}, Ground Truth: {true_class}")

    for rank in ["kingdom", "phylum", "class", "order", "family", "genus", "species"]:
        if rank not in predictions or rank not in ground_truth:
            continue  # ✅ Skip missing ranks

        pred_logits = predictions[rank]
        true_label = ground_truth[rank]

        # ✅ Filter valid labels (ignore entries where ground truth == -1)
        valid_mask = true_label != -1
        if valid_mask.sum() == 0:
            # print(f"[WARNING] No valid labels for rank '{rank}', skipping loss calculation.")
            continue  # ✅ Skip rank if all labels are invalid

        pred_logits = pred_logits[valid_mask]
        true_label = true_label[valid_mask]

        # ✅ Prevent index errors
        num_classes = pred_logits.shape[1]
        if (true_label < 0).any() or (true_label >= num_classes).any():
            print(
                f"[ERROR] Rank '{rank}' has out-of-bounds label indices! {true_label.tolist()}"
            )
            continue  # ✅ Skip invalid data to prevent crashes

        # ✅ Compute loss safely
        rank_loss = F.cross_entropy(pred_logits, true_label)

        # ✅ Compute hierarchical distance penalty
        pred_classes = pred_logits.argmax(dim=1).cpu().tolist()
        true_classes = true_label.cpu().tolist()

        taxonomic_distances = [
            compute_taxonomic_distance(true, pred, taxonomy_tree)
            for true, pred in zip(true_classes, pred_classes)
        ]
        avg_distance = (
            sum(taxonomic_distances) / len(taxonomic_distances)
            if taxonomic_distances
            else 0
        )

        total_loss += rank_loss + (distance_penalty * avg_distance)
        total_distance += avg_distance

    return total_loss / len(predictions), total_distance / len(predictions)


def compute_taxonomic_distance(true_class, pred_class, taxonomy_tree):
    """Computes the taxonomic distance between predicted and true labels."""
    true_rank = next(
        (rank for rank in taxonomy_tree if true_class in taxonomy_tree[rank]), None
    )
    pred_rank = next(
        (rank for rank in taxonomy_tree if pred_class in taxonomy_tree[rank]), None
    )

    if true_rank is None or pred_rank is None:
        return 12  # ✅ Assign max penalty if class not found

    rank_order = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
    return abs(rank_order.index(true_rank) - rank_order.index(pred_rank))


def compute_accuracy(predictions, ground_truth):
    """
    Computes per-rank accuracy across taxonomic classifications.

    Args:
        predictions (dict): Model outputs per rank.
        ground_truth (dict): True labels per rank.

    Returns:
        dict: Accuracy scores per taxonomy rank.
    """
    accuracy_per_rank = {}

    for rank in ["kingdom", "phylum", "class", "order", "family", "genus", "species"]:
        if rank not in predictions or rank not in ground_truth:
            accuracy_per_rank[rank] = None  # ✅ Skip missing ranks
            continue

        pred_classes = predictions[rank].argmax(dim=1)
        true_classes = ground_truth[rank]

        correct = (pred_classes == true_classes).sum().item()
        total = len(true_classes)

        accuracy_per_rank[rank] = (
            correct / total if total > 0 else 0.0
        )  # ✅ Avoid division by zero

    return accuracy_per_rank
