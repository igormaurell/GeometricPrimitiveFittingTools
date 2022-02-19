from copy import deepcopy
import numpy as np
import random

from lib.utils import computeFeaturesPointIndices, sortedIndicesIntersection

def compute3DRegionSize(region_size, region_axis):
    assert region_axis in ['x', 'y', 'z']
    index = ['x', 'y', 'z'].index(region_axis)
    size = list(region_size)
    size.insert(index, np.inf)
    size = np.array(size)
    return size

def computeGridOfRegions(points, region_size, region_axis):
    assert region_axis in ['x', 'y', 'z']
    index = ['x', 'y', 'z'].index(region_axis)

    size = np.array(region_size)

    min = np.min(points, 0)
    min = np.delete(min, index)
    max = np.max(points, 0)
    max = np.delete(max, index)
    points_size = max - min
    
    num_parts = np.ceil(points_size/size)
    num_parts = num_parts.astype('int64')

    regions = [[ None for j in range(num_parts[1])] for i in range(num_parts[0])]

    for i in range(num_parts[0]):
        m0 = min[0] + i*size[0]
        M0 = min[0] + (i+1)*size[0]
        for j in range(num_parts[1]):
            m1 = min[1] + j*size[1]
            M1 = min[1] + (j+1)*size[1]
            ll = [m0, m1]
            ll.insert(index, -np.inf)
            ll = np.array(ll)
            ur = [M0, M1]
            ur.insert(index, np.inf)
            ur = np.array(ur)
            regions[i][j] = (ll, ur)
        
    return regions

def computeRegionAroundPoint(point, region_size, region_axis):
    size = compute3DRegionSize(region_size, region_axis)

    ll = point - size/2
    ur = point + size/2

    return ll, ur

def randomSamplingPointsOnRegion(points, ll, ur, n_points):
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

def sampleDataOnRegion(region, points, normals, labels, features_data, region_size, region_axis, n_points,
                   filter_features_by_volume=True, abs_volume_threshold=0., relative_volume_threshold=0.2):
    
    ll, ur = region

    indices = randomSamplingPointsOnRegion(points, ll, ur, n_points)

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

def divideOnceRandom(points, normals, labels, features_data, region_size, region_axis, n_points,
                     filter_features_by_volume=True, abs_volume_threshold=0., relative_volume_threshold=0.2):
    
    middle_point = points[random.randint(0, points.shape[0])]

    region = computeRegionAroundPoint(middle_point, region_size, region_axis)

    return sampleDataOnRegion(region, points, normals, labels, features_data, region_size, region_axis, n_points, filter_features_by_volume=filter_features_by_volume,
                       abs_volume_threshold=abs_volume_threshold, relative_volume_threshold=relative_volume_threshold)
