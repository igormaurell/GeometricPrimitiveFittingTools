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
            filename = str(uuid.uuid4())
        
        self.filenames.append(filename)

        h5_file_path = os.path.join(self.folder_name, f'{filename}.h5')

        if os.path.exists(h5_file_path):
           return False

        print(h5_file_path)

        with h5py.File(h5_file_path, 'w') as h5_file:
            noise_limit = 0.
            if 'add_noise' in self.normalization_parameters.keys():
                noise_limit = self.normalization_parameters['add_noise']
                self.normalization_parameters['add_noise'] = 0.

            gt_points, gt_normals, features_data, transforms = normalize(points.copy(), self.normalization_parameters, normals=normals.copy(),features=features_data)

            h5_file.create_dataset('gt_points', data=gt_points)
            if gt_normals is not None:
                h5_file.create_dataset('gt_normals', data=gt_normals)

            del gt_normals
            gc.collect()

            if labels is not None:
                h5_file.create_dataset('gt_labels', data=labels)

            del labels
            gc.collect()

            if noise_limit != 0.:
                self.normalization_parameters['add_noise'] = noise_limit
                noisy_points, _, _, _ = normalize(points, self.normalization_parameters, normals=normals.copy())
                h5_file.create_dataset('noisy_points', data=noisy_points)
                del noisy_points

            del points
            gc.collect()

            point_position = h5_file_path.rfind('.')
            point_position = point_position if point_position >= 0 else len(point_position)
            bar_position = h5_file_path.rfind('/')
            bar_position = bar_position if bar_position >= 0 else 0

            for i, feature in enumerate(features_data):
                if len(feature['point_indices']) > 0:
                    soup_name = f'{filename}_soup_{i}'
                    grp = h5_file.create_group(soup_name)
                    points = gt_points[feature['point_indices']]
                    grp.create_dataset('gt_points', data=points)
                    feature['name'] = soup_name
                    feature['normalized'] = True
                    feature = filterFeature(feature, SpfnDatasetMaker.FEATURES_BY_TYPE, SpfnDatasetMaker.FEATURES_TRANSLATION)
                    grp.attrs['meta'] = np.void(pickle.dumps(feature))
    
    def finish(self):
        return True