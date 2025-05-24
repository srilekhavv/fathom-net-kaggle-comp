import torch.nn.functional as F


def hierarchical_loss(predictions, ground_truth, taxonomy_tree, distance_penalty=0.1):
    """
    Computes taxonomic-aware loss based on hierarchical distance.

    Args:
        predictions (dict): Model outputs for multiple taxonomic ranks.
        ground_truth (dict): True labels across taxonomic ranks.
        taxonomy_tree (dict): Reference taxonomy structure.
        distance_penalty (float): Weight for hierarchical distance penalty.

    Returns:
        torch.Tensor: Hierarchical penalty-based loss.
    """
    total_loss = 0
    total_distance = 0

    for rank in taxonomy_tree.keys():
        pred_logits = predictions[rank]  # Predicted scores
        true_label = ground_truth[rank]  # True labels

        # ✅ Debug print: Check tensor shapes before computing loss
        print(f"\n[DEBUG] Rank: {rank}")
        print(f"  [DEBUG] Prediction Logits Shape: {pred_logits.shape}")
        print(f"  [DEBUG] True Label Shape: {true_label.shape}")
        print(f"  [DEBUG] True Label Values: {true_label}")

        # ✅ Check for out-of-bounds labels before computing loss
        num_classes = pred_logits.shape[1]  # Expected number of classes
        if (true_label < 0).any() or (true_label >= num_classes).any():
            print(
                f"[ERROR] Rank '{rank}' has out-of-bounds label indices! {true_label.tolist()}"
            )

        # ✅ Standard cross-entropy loss
        rank_loss = F.cross_entropy(pred_logits, true_label)

        # ✅ Compute hierarchical distance penalty
        pred_classes = pred_logits.argmax(dim=1).cpu().tolist()
        true_classes = true_label.cpu().tolist()

        # ✅ Debug print before distance computation
        print(f"  [DEBUG] Predicted Classes: {pred_classes}")
        print(f"  [DEBUG] Ground Truth Classes: {true_classes}")

        taxonomic_distances = [
            compute_taxonomic_distance(true, pred, taxonomy_tree)
            for true, pred in zip(true_classes, pred_classes)
        ]

        avg_distance = sum(taxonomic_distances) / len(taxonomic_distances)
        total_loss += rank_loss + (
            distance_penalty * avg_distance
        )  # ✅ Penalize based on hierarchy
        total_distance += avg_distance

    return total_loss / len(taxonomy_tree), total_distance / len(taxonomy_tree)


def compute_taxonomic_distance(true_class, pred_class, taxonomy_tree):
    """
    Computes the taxonomic distance between predicted and true labels.

    If classes exist in different taxonomic ranks, traverses up the tree to calculate distance.

    Args:
        true_class (str): True classification.
        pred_class (str): Predicted classification.
        taxonomy_tree (dict): Full taxonomic hierarchy.

    Returns:
        int: Hierarchical distance between predictions.
    """
    # ✅ Find ranks of both classes
    true_rank = next(
        (r for r, classes in taxonomy_tree.items() if true_class in classes), None
    )
    pred_rank = next(
        (r for r, classes in taxonomy_tree.items() if pred_class in classes), None
    )

    # ✅ If either class isn't found, return max penalty
    if not true_rank or not pred_rank:
        print(
            f"[WARNING] Taxonomic mapping missing for true_class={true_class} or pred_class={pred_class}"
        )
        return 12

    # ✅ Compute hierarchical distance
    rank_order = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
    return abs(rank_order.index(true_rank) - rank_order.index(pred_rank))
