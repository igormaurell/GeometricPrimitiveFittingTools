import pickle
import h5py
import numpy as np
import gc
import uuid
import os

from lib.normalization import normalize
from lib.utils import filterFeature, computeLabelsFromFace2Primitive

from .base_dataset_maker import BaseDatasetMaker

class SpfnDatasetMaker(BaseDatasetMaker):
    FEATURES_BY_TYPE = {
        'plane': ['name', 'location', 'z_axis', 'normalized'],
        'cylinder': ['name', 'location', 'z_axis', 'radius', 'normalized'],
        'cone': ['name', 'location', 'z_axis', 'radius', 'angle', 'apex', 'normalized'],
        'sphere': ['name', 'location', 'radius', 'normalized']
    }

    FEATURES_TRANSLATION = {}

    def __init__(self, parameters):
        super().__init__(parameters)

    def step(self, points, normals=None, labels=None, features_data=[], filename=None):
        if filename is None:
            filename = str(uuid.uuid4())
        
        self.filenames.append(filename)

        data_file_path = os.path.join(self.data_folder_name, f'{filename}.h5')
        transforms_file_path = os.path.join(self.transform_folder_name, f'{filename}.pkl')

        features_data = features_data['surfaces']

        if os.path.exists(data_file_path):
           return False

        with h5py.File(data_file_path, 'w') as h5_file:
            noise_limit = 0.
            if 'add_noise' in self.normalization_parameters.keys():
                noise_limit = self.normalization_parameters['add_noise']
                self.normalization_parameters['add_noise'] = 0.

            gt_points, gt_normals, features_data, transforms = normalize(points.copy(), self.normalization_parameters, normals=normals.copy(), features=features_data)

            with open(transforms_file_path, 'wb') as pkl_file:
                pickle.dump(transforms, pkl_file)

            h5_file.create_dataset('gt_points', data=gt_points)
            if gt_normals is not None:
                h5_file.create_dataset('gt_normals', data=gt_normals)

            del gt_normals
            gc.collect()

            features_point_indices = []
            if labels is not None:
                labels, features_point_indices = computeLabelsFromFace2Primitive(labels, features_data)
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

            point_position = data_file_path.rfind('.')
            point_position = point_position if point_position >= 0 else len(point_position)
            bar_position = data_file_path.rfind('/')
            bar_position = bar_position if bar_position >= 0 else 0

            for i, feature in enumerate(features_data):
                if len(features_point_indices[i]) > 0:
                    soup_name = f'{filename}_soup_{i}'
                    grp = h5_file.create_group(soup_name)
                    points = gt_points[features_point_indices[i]]
                    grp.create_dataset('gt_points', data=points)
                    feature['name'] = soup_name
                    feature['normalized'] = True
                    feature = filterFeature(feature, SpfnDatasetMaker.FEATURES_BY_TYPE, SpfnDatasetMaker.FEATURES_TRANSLATION)
                    grp.attrs['meta'] = np.void(pickle.dumps(feature))
        return True

    def finish(self, permutation=None):
        train_models, test_models = self.divideTrainVal(permutation)
        with open(os.path.join(self.data_folder_name, 'train_models.csv'), 'w') as f:
            text = ','.join([f'{filename}.h5' for filename in train_models])
            f.write(text)
        with open(os.path.join(self.data_folder_name, 'test_models.csv'), 'w') as f:
            text = ','.join([f'{filename}.h5' for filename in test_models])
            f.write(text)