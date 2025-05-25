import torch.nn.functional as F


def hierarchical_loss(
    predictions, ground_truth, taxonomy_tree, label_mapping, distance_penalty=0.1
):
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
            compute_taxonomic_distance(true, pred, taxonomy_tree, label_mapping)
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


def compute_taxonomic_distance(
    true_label_idx, pred_label_idx, label_mapping, taxonomy_tree
):
    """Finds the highest divergence point and sums all mistakes in taxonomy path."""

    rank_order = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]

    # ✅ Decode label indices to taxonomic names
    true_label = next(
        (
            name
            for name, idx in label_mapping["species"].items()
            if idx == true_label_idx
        ),
        "UNKNOWN",
    )
    pred_label = next(
        (
            name
            for name, idx in label_mapping["species"].items()
            if idx == pred_label_idx
        ),
        "UNKNOWN",
    )

    if true_label == "UNKNOWN" or pred_label == "UNKNOWN":
        print(
            f"[WARNING] Label index not found: True={true_label_idx}, Pred={pred_label_idx}"
        )
        return 12  # ✅ Default max distance if decoding fails

    # ✅ Extract taxonomy paths
    true_taxonomy = taxonomy_tree.get(true_label, {})
    pred_taxonomy = taxonomy_tree.get(pred_label, {})

    # ✅ Find the highest point of divergence
    divergence_rank = None
    for rank in rank_order:
        if rank in true_taxonomy and rank in pred_taxonomy:
            if true_taxonomy[rank] == pred_taxonomy[rank]:  # ✅ Matching rank
                divergence_rank = rank
            else:
                break  # ✅ Stop at first mismatch
        else:
            break  # ✅ Stop if one taxon is missing this rank

    # ✅ Count mistakes **after the divergence point**
    true_mistakes = sum(
        1
        for r in rank_order
        if r in true_taxonomy
        and divergence_rank
        and rank_order.index(r) > rank_order.index(divergence_rank)
    )
    pred_mistakes = sum(
        1
        for r in rank_order
        if r in pred_taxonomy
        and divergence_rank
        and rank_order.index(r) > rank_order.index(divergence_rank)
    )

    return true_mistakes + pred_mistakes  # ✅ Total taxonomic error count


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
