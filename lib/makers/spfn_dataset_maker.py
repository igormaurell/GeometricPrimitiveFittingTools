from tqdm import tqdm

import pickle

import h5py

import numpy as np

import gc

from ..normalization import normalize

class SpfnDatasetMaker:
    FEATURES_BY_TYPE = {
        'plane': ['location', 'z_axis', 'normalized'],
        'cylinder': ['location', 'z_axis', 'radius', 'normalized'],
        'cone': ['location', 'z_axis', 'radius', 'angle', 'apex', 'normalized'],
        'sphere': ['location', 'radius', 'normalized']
    }

    FEATURES_TRANSLATION = {}

    def __init__(self, parameters):
        self.folder_name = parameters['folder_name']
        self.normalization_parameters = parameters['normalization']


def generateH52SPFN(point_cloud, h5_filename, labels = None, features_data = None, norm_parameters = None):
    with h5py.File(h5_filename, 'w') as h5_file:
        noise_limit = 0.
        if 'add_noise' in norm_parameters.keys():
            noise_limit = norm_parameters['add_noise']
            norm_parameters['add_noise'] = 0.

        gt_point_cloud, features_data, transforms = normalize(point_cloud.copy(), norm_parameters, features=features_data)

        h5_file.create_dataset('gt_points', data=gt_point_cloud[:, :3])
        h5_file.create_dataset('gt_normals', data=gt_point_cloud[:, 3:])
        points = gt_point_cloud[:, :3]

        del gt_point_cloud
        gc.collect()

        h5_file.create_dataset('gt_labels', data=labels)

        del labels
        gc.collect()

        norm_parameters['add_noise'] = noise_limit
        point_cloud, _, _ = normalize(point_cloud, norm_parameters)
        h5_file.create_dataset('noisy_points', data=point_cloud[:, :3])

        del point_cloud
        gc.collect()

        point_position = h5_filename.rfind('.')
        point_position = point_position if point_position >= 0 else len(point_position)
        bar_position = h5_filename.rfind('/')
        bar_position = bar_position if bar_position >= 0 else 0

        name = h5_filename[bar_position+1:point_position]

        for i, feature in enumerate(features_data):
            if len(feature['point_indices']) > 0:
                soup_name = f'{name}_soup_{i}'
                grp = h5_file.create_group(soup_name)
                gt_points = points[feature['point_indices']]
                grp.create_dataset('gt_points', data=gt_points)
                feature = filterFeature2SPFN(feature, soup_name)
                grp.attrs['meta'] = np.void(pickle.dumps(feature))