import numpy as np
import random

from lib.utils import computeFeaturesPointIndices, sortedIndicesIntersection

def computeRegionAroundPoint(point, region_size, region_axis):
    assert region_axis in ['x', 'y', 'z']
    if region_axis == 'x':
        size = np.array([np.inf, region_size[0], region_size[1]])
    elif region_axis == 'y':
        size = np.array([region_size[0], np.inf, region_size[1]])
    else:
        size = np.array([region_size[0], region_size[1], np.inf])
    
    ll = point - size/2
    ur = point + size/2

    return ll, ur

def randomSamplingOnRegion(points, ll, ur, n_points):
    inidx = np.all(np.logical_and(points >= ll, points <= ur), axis=1)
    indices = np.arange(0, inidx.shape[0], 1, dtype=int)
    indices = indices[inidx]

    perm = np.random.permutation(indices.shape[0])
    indices = indices[perm[:n_points]]
    indices.sort()

    return indices

def featuresIndicesByPointsIndices(features_point_indices, points_indices, filter_by_volume=True, points=None, abs_volume_threshold=0., relative_volume_threshold=0.2):
    features_indices = []
    if filter_by_volume and points is None:
        print('Parameter points is need when filter_by_volume=True. The full point cloud must be used.')
    for i, fpi in enumerate(features_point_indices):
        fpi.sort()
        keep_fpi = sortedIndicesIntersection(fpi, points_indices)
        if len(keep_fpi) > 0:
            if filter_by_volume and points is not None:
                points_of_feature = points[fpi]
                volume_of_feature = np.linalg.norm((np.max(points_of_feature, 0) - np.min(points_of_feature, 0)), ord=2)
                
                if volume_of_feature > abs_volume_threshold:
                    points_of_feature_crop = points[keep_fpi]
                    volume_of_feature_crop = np.linalg.norm((np.max(points_of_feature_crop, 0) - np.min(points_of_feature_crop, 0)), ord=2)

                    volume_relative = volume_of_feature_crop/volume_of_feature

                    if volume_relative > relative_volume_threshold and volume_of_feature_crop > abs_volume_threshold:
                        features_indices.append(i)
            else:
                features_indices.append(i)

    return np.array(features_indices)



def divideOnceRandom(points, normals, labels, features_data, region_size, region_axis, n_points,
                                    filter_features_by_volume=True, abs_volume_threshold=0., relative_volume_threshold=0.2):
    middle_point = points[random.randint(0, points.shape[0])]

    ll, ur = computeRegionAroundPoint(middle_point, region_size, region_axis)

    indices = randomSamplingOnRegion(points, ll, ur, n_points)

    points_part = points[indices]
    normals_part = normals[indices]
    labels_part = labels[indices]

    features_point_indices = computeFeaturesPointIndices(labels)

    features_indices = featuresIndicesByPointsIndices(features_point_indices, indices, filter_by_volume=filter_features_by_volume, points=points,
                                                      abs_volume_threshold=abs_volume_threshold, relative_volume_threshold=relative_volume_threshold)

    features_data_part = [None for f in features_indices]
    labels_part_new = np.zeros(len(labels_part), dtype=np.int64) - 1
    for i, index in enumerate(features_indices):
        features_data_part[i] = features_data[index]
        labels_part_new[labels_part == index] = i
    features_data = features_data_part
    labels_part = labels_part_new

    result = {
        'points': points_part,
        'normals': normals_part,
        'labels': labels_part,
        'features': features_data_part,
    }

    return result