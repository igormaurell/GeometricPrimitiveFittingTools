import numpy as np
#from lapsolver import solve_dense

# def hungarian_matching(W_pred, W_gt):
#     # This non-tf function does not backprob gradient, only output matching indices
#     # W_pred - NxK
#     # W_gt - NxK'
#     # Output: matching_indices
#     # The matching does not include gt background instance
#     # calculate RIoU
#     n_points = W_pred.shape[0]
#     #n_max_labels = min(W_gt.shape[1], W_pred.shape[1])
#     #matching_indices = np.zeros([n_max_labels], dtype=np.int32)

#     dot = np.sum(np.expand_dims(W_pred, axis=2) * np.expand_dims(W_gt, axis=1),
#                  axis=0)  # K'xK
#     denominator = np.expand_dims(np.sum(W_pred, axis=0),
#                                  axis=1) + np.expand_dims(np.sum(W_gt, axis=0),
#                                                           axis=0) - dot
#     cost = dot / np.maximum(denominator, np.finfo(np.float64).eps)  # K'xK
#     row_ind, col_ind = solve_dense(-cost)  # want max solution
#     #matching_indices[b, :n_gt_labels] = col_ind

#     return row_ind, col_ind

# def compute_riou(W_pred, W_gt, pred_ind, gt_ind):
#     # W_pred - NxK
#     # W_gt - NxK'

#     N, _ = W_pred.shape

#     pred_ind = torch.LongTensor(pred_ind).unsqueeze(0).repeat(N, 1).to(W_pred.device)
#     gt_ind = torch.LongTensor(gt_ind).unsqueeze(0).repeat(N, 1).to(W_gt.device)

#     W_pred_reordered = torch.gather(W_pred, -1, pred_ind)
#     W_gt_reordered = torch.gather(W_gt, -1, gt_ind)

#     dot = torch.sum(W_gt_reordered * W_pred_reordered, dim=0)  # K
#     denominator = torch.sum(W_gt_reordered, dim=0) + torch.sum(
#         W_pred_reordered, dim=0) - dot
#     mIoU = dot / (denominator + DIVISION_EPS)  # K
#     return mIoU

# def compute_miou(cluster_pred, I_gt):
#     '''
#     compute per-primitive riou loss
#     cluster_pred: (1, N)
#     I_gt: (1, N), must contains -1
#     '''
#     assert (cluster_pred.shape[0] == 1)

#     one_hot_pred = get_one_hot(cluster_pred,
#                                cluster_pred.max() + 1)[0]  # (N, K)

#     if I_gt.min() == -1:
#         one_hot_gt = get_one_hot(I_gt + 1,
#                                  I_gt.max() +
#                                  2)[0][:, 1:]  # (N, K'), remove background
#     else:
#         one_hot_gt = get_one_hot(I_gt, I_gt.max() + 1)[0]

#     pred_ind, gt_ind = hungarian_matching(npy(one_hot_pred), npy(one_hot_gt))
#     riou = compute_riou(one_hot_pred, one_hot_gt, pred_ind, gt_ind)
#     k = riou.shape[0]
#     mean_riou = riou.sum() / k
#     return mean_riou, pred_ind, gt_ind


# def compute_type_miou(type_per_point, T_gt, cluster_pred, I_gt):

#     assert (type_per_point.shape[0] == 1)
    
#     # get T_pred: (1, N)
#     if len(type_per_point.shape) == 3:
#         B, N, _ = type_per_point.shape
#         T_pred = torch.argmax(type_per_point, dim=-1) # (B, N)
#     else:
#         T_pred = type_per_point

   
#     one_hot_pred = get_one_hot(cluster_pred,
#                                cluster_pred.max() + 1)[0]  # (N, K)

#     if I_gt.min() == -1:
#         # (N, K'), remove background
#         one_hot_gt = get_one_hot(I_gt + 1,
#                                  I_gt.max() + 2)[0][:, 1:]  
#     else:
#         one_hot_gt = get_one_hot(I_gt, I_gt.max() + 1)[0]

#     pred_ind, gt_ind = hungarian_matching(npy(one_hot_pred), npy(one_hot_gt))
#     type_iou = torch.Tensor([0.0]).to(T_gt.device)
#     cnt = 0
    
#     for p_ind, g_ind in zip(pred_ind, gt_ind):
#         gt_type_label = T_gt[I_gt == g_ind].mode()[0]
#         pred_type_label = T_pred[cluster_pred == p_ind].mode()[0]
#         if gt_type_label == pred_type_label:
#             type_iou += 1
#         cnt += 1
    
#     type_iou /= cnt
#     return type_iou

def computeIoUs(query_labels, gt_labels, p=False):
    ious = [None] * (np.max(query_labels) + 1)

    intersections = []

    for i in range(len(ious)):
        query_mask = (query_labels == i)
        gt_mask = (gt_labels == i)
        intersection = np.count_nonzero(np.logical_and(query_mask, gt_mask))
        union = np.count_nonzero(np.logical_or(query_mask, gt_mask))
        if intersection > 0:
            intersections.append((i, intersection, union))
        ious[i] = intersection/(union + np.finfo(np.float64).eps)

    if p:
        print(intersections)

    return ious
