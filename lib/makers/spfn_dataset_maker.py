import pickle
import h5py
import numpy as np
import gc
import uuid
import os
from lib.normalization import normalize
from lib.utils import filterFeature

class SpfnDatasetMaker:
    FEATURES_BY_TYPE = {
        'plane': ['name', 'location', 'z_axis', 'normalized'],
        'cylinder': ['name', 'location', 'z_axis', 'radius', 'normalized'],
        'cone': ['name', 'location', 'z_axis', 'radius', 'angle', 'apex', 'normalized'],
        'sphere': ['name', 'location', 'radius', 'normalized']
    }

    FEATURES_TRANSLATION = {}

    def __init__(self, parameters):
        self.folder_name = parameters['folder_name']
        self.normalization_parameters = parameters['normalization']
        self.filenames = []

    def step(self, points, normals=None, labels=None, features_data=[], filename=None):
        if filename is None:
            filename = f'{str(uuid.uuid4())}.h5'
        else:
            filename = f'{filename}.h5'
        
        self.filenames.append(filename)

        h5_file_path = os.path.join(self.folder_name, filename)

        if os.path.exists(h5_file_path):
           return False

        with h5py.File(h5_file_path, 'w') as h5_file:
            noise_limit = 0.
            if 'add_noise' in self.normalization_parameters.keys():
                noise_limit = self.normalization_parameters['add_noise']
                self.normalization_parameters['add_noise'] = 0.

            gt_points, gt_normals, features_data, transforms = normalize(points.copy(), normals.copy(), self.normalization_parameters, features=features_data)

            h5_file.create_dataset('gt_points', data=gt_points)
            h5_file.create_dataset('gt_normals', data=gt_normals)

            del gt_normals
            gc.collect()

            if labels is not None:
                h5_file.create_dataset('gt_labels', data=labels)

            del labels
            gc.collect()

            self.normalization_parameters['add_noise'] = noise_limit
            noisy_points, _, _, _ = normalize(points, normals, self.normalization_parameters)
            h5_file.create_dataset('noisy_points', data=noisy_points)

            del noisy_points
            del points
            gc.collect()

            point_position = h5_file_path.rfind('.')
            point_position = point_position if point_position >= 0 else len(point_position)
            bar_position = h5_file_path.rfind('/')
            bar_position = bar_position if bar_position >= 0 else 0

            name = h5_file_path[bar_position+1:point_position]

            for i, feature in enumerate(features_data):
                if len(feature['point_indices']) > 0:
                    soup_name = f'{name}_soup_{i}'
                    grp = h5_file.create_group(soup_name)
                    points = gt_points[feature['point_indices']]
                    grp.create_dataset('gt_points', data=gt_points)
                    feature['name'] = soup_name
                    feature['normalized'] = True
                    feature = filterFeature(feature, SpfnDatasetMaker.FEATURES_BY_TYPE, SpfnDatasetMaker.FEATURES_TRANSLATION)
                    grp.attrs['meta'] = np.void(pickle.dumps(feature))