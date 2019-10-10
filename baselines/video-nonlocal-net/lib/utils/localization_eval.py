import numpy as np


def l1_dist_labels(labels, preds):
    """Compute the L1 (manhattan) distance between the prediction and labels.
    Use the grid structure based on the size of the labels/preds.
    **ASSUMPTION** That the grid is a square, so the number of classes gives away
    the sides, and hence the structure of the grid.
    Args:
        labels: (N,) for softmax case, (N, C) for sigmoid case
        preds: (N, C)
    """
    grid_size = np.sqrt(preds.shape[1])
    assert grid_size * grid_size == preds.shape[1]
    assert len(labels.shape) == 1
    assert labels.shape[0] == preds.shape[0]
    x_lbl = labels % grid_size
    y_lbl = labels // grid_size
    x_pred = np.argmax(preds, axis=1) % grid_size
    y_pred = np.argmax(preds, axis=1) // grid_size
    dist = np.abs(x_lbl - x_pred) + np.abs(y_lbl - y_pred)
    return np.mean(dist, axis=0)


def compute_top_k_acc(labels, preds, k):
    """Compute accuracy.
    Args:
        labels: (N,) for softmax case, (N, C) for sigmoid case
        preds: (N, C)
    """
    top_k_preds = np.fliplr(np.argsort(preds, -1))[:, :k]
    labels = np.tile(np.expand_dims(labels, 1), (1, k))
    any_match = np.any(top_k_preds == labels, axis=-1)
    return np.mean(any_match)


def all_localization_accuracies(labels, preds):
    return {
        'top_1': compute_top_k_acc(labels, preds, 1),
        'top_5': compute_top_k_acc(labels, preds, 5),
        'l1_dist': l1_dist_labels(labels, preds),
    }