import numpy as np

def mergeQueryAndGTData(query, gt, global_min=-1, num_points=0):
    if 'gt_indices' in query:
        gt_indices = np.array(query['gt_indices']) + num_points
    else:
        gt_indices = np.arange(len(query['points'])) + num_points
    
    assert 'matching' in query, 'hungarian matching not implemented yet'

    gt_labels = gt['labels']
    gt_labels_unique = np.unique(gt_labels)
    gt_labels_unique = gt_labels_unique[gt_labels_unique != -1]

    matching = np.array(query['matching'])
    valid_matching_mask = matching != -1
    matching[valid_matching_mask] = gt_labels_unique[matching[valid_matching_mask]]

    local_min = global_min
    for i in range(0, len(matching)):
        if matching[i] == -1:
            matching[i] = local_min - 1
            local_min -= 1
    
    new_labels = query['labels']
    valid_labels_mask = new_labels != -1

    new_labels[valid_labels_mask] = matching[new_labels[valid_labels_mask]]
    #print(np.unique(new_labels))

    old_features = query['features']
    new_features = [None]*(len(gt['features']))
    non_gt_features = [None]*np.count_nonzero(matching < global_min)
    for query_idx, gt_idx in enumerate(matching):
        if gt_idx < global_min:
            non_gt_features[gt_idx - global_min + 1] = old_features[query_idx]
        else:
            new_features[gt_idx] = old_features[query_idx]
    
    query['labels'] = new_labels
    query['features'] = new_features
    query['non_gt_features'] = non_gt_features
    query['gt_indices'] = gt_indices

    return query