import numpy as np
from lapsolver import solve_dense
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss

def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw

# from Mask3D
def memoryEfficientHungarianMatching(query_labels, gt_labels):
    """More memory-friendly matching"""
    num_queries = len(query_labels)

    out_mask = torch.as_tensor(getOneHot(query_labels), dtype=torch.int64)
    tgt_mask = torch.as_tensor(getOneHot(gt_labels), dtype=torch.int64)

    out_mask = out_mask.float()
    tgt_mask = tgt_mask.float()
    # Compute the focal loss between masks
    cost_mask = batch_sigmoid_ce_loss(out_mask, tgt_mask)

    # Compute the dice loss betwen masks
    cost_dice = batch_dice_loss(out_mask, tgt_mask)

    # Final cost matrix
    C = (cost_mask + cost_dice)
    print(C.shape)
    C = C.reshape(num_queries, -1)

    indices = linear_sum_assignment(C)

    print(indices)

    return [
        (
            torch.as_tensor(i, dtype=torch.int64),
            torch.as_tensor(j, dtype=torch.int64),
        )
        for i, j in indices
    ]

def getOneHot(a):
    if a.min() == -1:
        b = np.zeros((a.size, a.max() + 2), np.int8)
        b[np.arange(a.size), (a + 1)] = 1
        return b[:, 1:]
    else:
        b = np.zeros((a.size, a.max() + 1), np.int8)
        b[np.arange(a.size), a] = 1
        return b
    
# from HPNet
def hungarianMatching(query_labels, gt_labels):
    W_pred = getOneHot(query_labels)
    W_gt = getOneHot(gt_labels)

    dot = np.sum(np.expand_dims(W_pred, axis=2) * np.expand_dims(W_gt, axis=1),
                 axis=0)  # K'xK
    denominator = np.expand_dims(np.sum(W_pred, axis=0),
                                 axis=1) + np.expand_dims(np.sum(W_gt, axis=0),
                                                          axis=0) - dot
    cost = dot / np.maximum(denominator, np.finfo(np.float64).eps)  # K'xK
    pred_ind, gt_ind = solve_dense(-cost)  # want max solution

    matching = np.zeros(max(0, np.max(query_labels) + 1), dtype=np.int32) - 1
    if len(pred_ind) > 0:
        matching[pred_ind] = gt_ind

    return matching

def mergeQueryAndGTData(query, gt, global_min=-1, num_points=0):
    if 'gt_indices' in query:
        gt_indices = np.array(query['gt_indices'])
    else:
        gt_indices = np.arange(len(query['points']))

    global_indices = None
    if 'global_indices' in query and 'global_indices' in gt:
        # gt and query are merged by dataset_merger 
        query_gi = query['global_indices']
        gt_gi = gt['global_indices']
        assert np.all(np.isin(query_gi, gt_gi)), 'query is not in gt (you are maybe using different dataset versions)'
        global_map = np.zeros(np.max(gt_gi) + 1, dtype=np.int32) - 1
        global_map[gt_gi] = np.arange(len(gt_gi), dtype=np.int32)
        gt_indices = global_map[query_gi]
    elif 'global_indices' in query:
        # gt is global (without division step)
        global_indices = query['global_indices']
        gt_indices = global_indices
    elif 'global_indices' in gt:
        # just gt has global indices
        global_indices = gt['global_indices'][gt_indices]    

    assert np.allclose(query['points'], gt['points'][gt_indices], rtol=0., atol=1e-5), f"{query['points']} != {gt['points'][gt_indices]}"

    query_labels = query['labels']
    gt_labels = gt['labels'][gt_indices]

    #print(np.unique(query_labels))
    #print(np.unique(gt_labels))

    query_local = np.zeros(len(query_labels), dtype=np.int32) - 1
    valid_query = query_labels > -1
    _, query_local[valid_query] = np.unique(query_labels[valid_query], return_inverse=True)

    gt_local = np.zeros(len(gt_labels), dtype=np.int32) - 1
    valid_gt = gt_labels > -1
    gt_map, gt_local[valid_gt] = np.unique(gt_labels[valid_gt], return_inverse=True) #possible problem in gt_map

    if 'matching' not in query:
        matching = hungarianMatching(query_local, gt_local)
    else:
        matching = np.array(query['matching'])

    valid_matching_mask = matching != -1
    matching[valid_matching_mask] = gt_map[matching[valid_matching_mask]]

    local_min = global_min
    for i in range(0, len(matching)):
        if matching[i] == -1:
            matching[i] = local_min - 1
            local_min -= 1
        
    new_labels = query_local.copy()
    #new_labels[new_labels > len(gt_map)] = -1
    valid_labels_mask = new_labels != -1
    new_labels[valid_labels_mask] = matching[new_labels[valid_labels_mask]]    

    old_features = query['features_data']
    new_features = [None]*(len(gt['features_data']))
    non_gt_features = [None]*np.count_nonzero(matching < global_min)
    for query_idx, gt_idx in enumerate(matching):
        if gt_idx < global_min:
            non_gt_features[global_min - gt_idx - 1] = old_features[query_idx]
        else:
            new_features[gt_idx] = old_features[query_idx]
    
    query['labels'] = new_labels
    query['features_data'] = new_features
    query['non_gt_features'] = non_gt_features
    query['gt_indices'] = gt_indices + num_points
    if global_indices is not None:
       query['global_indices'] = global_indices


    return query