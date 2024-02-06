from copy import deepcopy
import numpy as np
import random

from lib.utils import sortedIndicesIntersection

EPS = np.finfo(np.float32).eps

def getRegionAxisMinMax(region_index, axis_min, axis_max, axis_size):
    if axis_size == np.inf:
        M = axis_max
        m = axis_min
    else:
        M = axis_min + (region_index+1)*axis_size
        m = M - axis_size

    return max(m, axis_min), min(M, axis_max)

def computeGridOfRegions(points, region_size):
    min_points = np.min(points, 0)
    max_points = np.max(points, 0)
    points_size = max_points - min_points

    num_parts = np.ceil(points_size/region_size)
    num_parts = num_parts.astype('int64')
    num_parts[num_parts==0] = 1

    #adapting regions size to current model
    rs = points_size/num_parts

    min_points -= EPS
    max_points += EPS

    regions = np.ndarray((num_parts[0], num_parts[1], num_parts[2], 2, 3), dtype=np.float64)

    for i in range(num_parts[0]):
        m0, M0 = getRegionAxisMinMax(i, min_points[0], max_points[0], rs[0])
        for j in range(num_parts[1]):
            m1, M1 = getRegionAxisMinMax(j, min_points[1], max_points[1], rs[1])
            for k in range(num_parts[2]):
                m2, M2 = getRegionAxisMinMax(k, min_points[2], max_points[2], rs[2])
                regions[i, j, k, 0, :] = np.array([m0, m1, m2])
                regions[i, j, k, 1, :] = np.array([M0, M1, M2])

    return regions

def computeSearchPoints(points, region_size):
    inf_axes = (region_size == np.inf)
    rs = region_size.copy()
    rs[inf_axes] = 0
    min_points = np.min(points, axis=0) + rs/2
    max_points = np.max(points, axis=0) - rs/2

    error_indices = min_points >= max_points
    
    min_points[error_indices] -= rs[error_indices]/2
    max_points[error_indices] += rs[error_indices]/2

    return points[np.all(np.logical_and(points >= min_points, points < max_points), axis=1)]

def computeRegionAroundPoint(point, region_size):
    bb_min_limit = point - region_size/2
    bb_max_limit = point + region_size/2

    return np.asarray([bb_min_limit, bb_max_limit])

import time
def randomSamplingPointsOnRegion(points, ll, ur, n_points):
    inidx = np.all(np.logical_and(points >= ll, points < ur), axis=1)
    indices = np.arange(0, inidx.shape[0], 1, dtype=int)
    indices = indices[inidx]
    
    if n_points > 0 and indices.shape[0] > n_points:
        perm = np.random.permutation(indices.shape[0])
        indices = indices[perm[:n_points]]

    indices.sort()

    return indices

def featuresIndicesByPointsIndices(features_point_indices, points_indices):
    features_indices = []
    for i, fpi in enumerate(features_point_indices):
        fpi.sort()
        keep_fpi = sortedIndicesIntersection(fpi, points_indices)
        if len(keep_fpi) > 0:
            features_indices.append(i)
    return np.array(features_indices)

def sampleDataOnRegion(region, data, n_points):
    ll = region[0, :]
    ur = region[1, :]

    points = data['points']
    noisy_points = data['noisy_points']
    normals = data['normals']
    noisy_normals = data['noisy_normals']
    labels = data['labels']
    if labels is None:
        labels = np.zeros(points.shape[0], dtype=np.int32) - 1
    features_data = data['features_data']

    indices = randomSamplingPointsOnRegion(points, ll, ur, n_points)

    if len(indices) == 0:
        return {
            'points': np.array([]),
            'normals': np.array([]),
            'labels': np.array([]),
            'global_indices': np.array([]),
            'features_data': [],
        }

    points_part = points[indices]
    noisy_points_part = noisy_points[indices]
    normals_part = normals[indices]
    noisy_normals_part = noisy_normals[indices]
    labels_part = labels[indices]

    features_indices = np.unique(labels_part)
    if features_indices[0] == -1:
        features_indices = features_indices[1:]

    
    features_data_part = [None] * len(features_data)
    for fi in features_indices:
        features_data_part[fi] = features_data[fi]

    result = {
        'points': points_part,
        'noisy_points': noisy_points_part,
        'normals': normals_part,
        'noisy_normals': noisy_normals_part,
        'labels': labels_part,
        'global_indices': indices,
        'features_data': features_data_part,
    }

    return result

def divideOnceRandom(data, region_size, n_points, search_points=None):
    points = data['points']

    if search_points is None:
        middle_point = points[random.randint(0, points.shape[0] - 1)]
    else:
        middle_point = search_points[random.randint(0, search_points.shape[0] - 1)]

    region = computeRegionAroundPoint(middle_point, region_size)

    return sampleDataOnRegion(region, data, n_points)
