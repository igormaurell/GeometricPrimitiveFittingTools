from tqdm import tqdm

import pickle

import h5py

import numpy as np

import gc

from normalization import normalize

SPFN_FEATURES = {
    'plane': ['location', 'z_axis', 'normalized'],
    'cylinder': ['location', 'z_axis', 'radius', 'normalized'],
    'cone': ['location', 'z_axis', 'radius', 'angle', 'apex', 'normalized'],
    'sphere': ['location', 'radius', 'normalized']
}

def filterFeature2SPFN(feature, name):
    tp = feature['type'].lower()
    if tp in SPFN_FEATURES.keys():
        feature_out = {}
        feature_out['name'] = name
        feature_out['type'] = tp
        for field in SPFN_FEATURES[tp]:
            feature_out[field] = feature[field]
        return feature_out

    feature['name'] = name
    feature['type'] = tp
    return feature

def generateH52SPFN(point_cloud, labels, features_data, h5_filename, parameters):
    h5_file = h5py.File(h5_filename, 'w')

    noise_limit = 0.
    if 'add_noise' in parameters.keys():
        noise_limit = parameters['add_noise']
        parameters['add_noise'] = 0.

    gt_point_cloud, features_data = normalize(point_cloud.copy(), features=features_data)

    h5_file.create_dataset('gt_points', data=gt_point_cloud[:, :3])
    h5_file.create_dataset('gt_normals', data=gt_point_cloud[:, 3:])
    points = gt_point_cloud[:, :3]

    del gt_point_cloud
    gc.collect()

    h5_file.create_dataset('gt_labels', data=labels)

    del labels
    gc.collect()

    parameters['add_noise'] = noise_limit
    point_cloud, _ = normalize(point_cloud, parameters)
    h5_file.create_dataset('noisy_points', data=point_cloud[:, :3])

    del point_cloud
    gc.collect()

    point_position = h5_filename.rfind('.')
    point_position = point_position if point_position >= 0 else len(point_position)
    bar_position = h5_filename.rfind('/')
    bar_position = bar_position if bar_position >= 0 else 0

    name = h5_filename[bar_position+1:point_position]

    for i, feature in enumerate(features_data):
        soup_name = f'{name}_soup_{i}'
        grp = h5_file.create_group(soup_name)
        gt_points = points[feature['point_indices'].sort()]
        grp.create_dataset('gt_points', data=gt_points)
        feature = filterFeature2SPFN(feature, soup_name)
        grp.attrs['meta'] = np.void(pickle.dumps(feature))

    h5_file.close()