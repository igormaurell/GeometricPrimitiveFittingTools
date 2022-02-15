import pickle
import h5py
import numpy as np
import gc
import uuid
import os

from lib.normalization import normalize
from lib.utils import filterFeature

from .base_dataset_reader import BaseDatasetReader

class DefaultDatasetReader(BaseDatasetReader):

    def __init__(self, parameters):
        super().__init__(parameters)

    def step(self, filename):
        self.filenames.append(filename)

        data_file_path = os.path.join(self.data_folder_name, f'{filename}.h5')
        transforms_file_path = os.path.join(self.transform_folder_name, f'{filename}.pkl')

        with h5py.File(data_file_path, 'r') as h5_file:
            noise_limit = 0.
            if 'add_noise' in self.normalization_parameters.keys():
                noise_limit = self.normalization_parameters['add_noise']
                self.normalization_parameters['add_noise'] = 0.

            gt_points, gt_normals, features_data, transforms = normalize(points.copy(), self.normalization_parameters, normals=normals.copy(),features=features_data)

            with open(transforms_file_path, 'wb') as pkl_file:
                pickle.dump(transforms, pkl_file)

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

            point_position = data_file_path.rfind('.')
            point_position = point_position if point_position >= 0 else len(point_position)
            bar_position = data_file_path.rfind('/')
            bar_position = bar_position if bar_position >= 0 else 0

            for i, feature in enumerate(features_data):
                if len(feature['point_indices']) > 0:
                    soup_name = f'{filename}_soup_{i}'
                    grp = h5_file.create_group(soup_name)
                    grp.create_dataset('gt_indices', data=feature['point_indices'])
                    feature['name'] = soup_name
                    feature['normalized'] = True
                    feature = filterFeature(feature, DefaultDatasetReader.FEATURES_BY_TYPE, DefaultDatasetReader.FEATURES_TRANSLATION)
                    grp.attrs['meta'] = np.void(pickle.dumps(feature))
        return True