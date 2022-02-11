from tqdm import tqdm

from os import listdir, mkdir, system
from os.path import isfile, join, exists

import pickle

import h5py

import numpy as np

from pypcd import pypcd

import gc

from normalization import centralize, add_noise

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

def normalize2SPFN(point_cloud, features=[], noise_limit = 10):
    point_cloud, features = centralize(point_cloud, features)
    if noise_limit != 0:
       point_cloud = add_noise(point_cloud, limit=noise_limit)
    for i in range(0, len(features)):
        features[i]['normalized'] = True
    return point_cloud, features


def generateH52SPFN(point_cloud, labels, features_data, h5_filename, parameters):
    h5_file = h5py.File(h5_filename, 'w')

    gt_point_cloud, features_data = normalize2SPFN(point_cloud.copy(), features=features_data, noise_limit=0)

    h5_file.create_dataset('gt_points', data=gt_point_cloud[:, :3])
    h5_file.create_dataset('gt_normals', data=gt_point_cloud[:, 3:])

    del gt_point_cloud
    gc.collect()

    h5_file.create_dataset('gt_labels', data=labels)

    del labels
    gc.collect()

    noisy_point_cloud, _ = normalize2SPFN(point_cloud, noise_limit=noise_limit) 

    h5_file.create_dataset('noisy_points', data=noisy_point_cloud[:, :3])

    del noisy_point_cloud
    gc.collect()

    point_position = h5_filename.rfind('.')
    point_position = point_position if point_position >= 0 else len(point_position)
    bar_position = h5_filename.rfind('/')
    bar_position = bar_position if bar_position >= 0 else 0

    name = h5_filename[bar_position+1:point_position]

    for i, feature in enumerate(features_data):
        soup_name = f'{name}_soup_{i}'
        grp = h5_file.create_group(soup_name)
        gt_indices = np.array(features_points[i])
        grp.create_dataset('gt_indices', data=gt_indices)
        feature = filterFeature2SPFN(feature, soup_name)
        grp.attrs['meta'] = np.void(pickle.dumps(feature))

    h5_file.close()

def generate2SPFN(pc, features_data, h5_filename, parameters):
    filterFeaturesData(features_data, parameters['curve_types'], parameters['surface_types'])
    generateH52SPFN(pc, features_data['surfaces'], h5_filename, parameters)