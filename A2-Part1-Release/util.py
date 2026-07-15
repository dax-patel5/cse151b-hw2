import numpy as np
import torch

def iou_components(pred, target, n_classes=21):
    """
    Per-class intersection and union pixel counts for a batch.

    Accumulate these across an entire dataset, then divide, to get IoU
    "determined over the whole validation set" rather than per batch.

    Args:
        pred (tensor): Predicted class labels, any shape.
        target (tensor): Ground truth labels, same shape as pred.
        n_classes (int): Number of classes.

    Returns:
        (np.ndarray, np.ndarray): intersection and union counts per class.
    """
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    intersection = np.zeros(n_classes, dtype=np.int64)
    union = np.zeros(n_classes, dtype=np.int64)
    for c in range(n_classes):
        p = pred == c
        t = target == c
        intersection[c] = (p & t).sum().item()
        union[c] = (p | t).sum().item()
    return intersection, union


def iou(pred, target, n_classes = 21):
    """
    Calculate the Intersection over Union (IoU) for predictions.

    Args:
        pred (tensor): Predicted output from the model.
        target (tensor): Ground truth labels.
        n_classes (int, optional): Number of classes. Default is 21.

    Returns:
        float: Mean IoU across all classes.
    """

    intersection, union = iou_components(pred, target, n_classes)
    ious = np.full(n_classes, np.nan)
    present = union > 0
    ious[present] = intersection[present] / union[present]
    # classes absent from both prediction and target don't contribute
    return float(np.nanmean(ious))


def pixel_acc(pred, target):
    """
    Calculate pixel-wise accuracy between predictions and targets.

    Args:
        pred (tensor): Predicted output from the model.
        target (tensor): Ground truth labels.

    Returns:
        float: Pixel-wise accuracy.
    """

    correct = (pred == target).sum().item()
    total = target.numel()
    return correct / total
