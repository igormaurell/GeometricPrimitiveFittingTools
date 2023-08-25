import numpy as np

def computeIoUs(query_labels, gt_labels):
    ious = [None] * (np.max(query_labels) + 1)

    for i in range(len(ious)):
        query_mask = (query_labels == i)
        gt_mask = (gt_labels == i)
        intersection = np.count_nonzero(np.logical_and(query_mask, gt_mask))
        union = np.count_nonzero(np.logical_or(query_mask, gt_mask))
        ious[i] = intersection/np.maximum(union, np.finfo(np.float64).eps)

    return ious
