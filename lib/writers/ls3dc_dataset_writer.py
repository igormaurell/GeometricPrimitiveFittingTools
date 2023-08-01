import pickle
import h5py
import csv
import uuid
import os
from collections.abc import Iterable

from lib.normalization import normalize
from lib.utils import filterFeaturesData, computeFeaturesPointIndices

from .base_dataset_writer import BaseDatasetWriter

def dict_to_hdf5_group(group, data_dict):
    for key, value in data_dict.items():
        if isinstance(value, dict):
            sub_group = group.create_group(key)
            dict_to_hdf5_group(sub_group, value)
        elif isinstance(value, Iterable) and not isinstance(value, str):
            sub_group = group.create_dataset(key, data=list(value))
        else:
            group[key] = value

import time
class LS3DCDatasetWriter(BaseDatasetWriter):
    def __init__(self, parameters):
        super().__init__(parameters)

    def step(self, points, normals=None, labels=None, features_data=[], noisy_points=None, filename=None, features_point_indices=None, **kwargs):
        import time

        if filename is None:
            filename = str(uuid.uuid4())
        
        data_file_path = os.path.join(self.data_folder_name, f'{filename}.h5')
        transforms_file_path = os.path.join(self.transform_folder_name, f'{filename}.pkl')

        if type(features_data) == dict:
            features_data = features_data['surfaces']

        if os.path.exists(data_file_path):
           return False

        if labels is not None:
            if features_point_indices is None:
                features_point_indices = computeFeaturesPointIndices(labels, size=len(features_data))

            min_number_points = self.min_number_points if self.min_number_points >= 1 else int(len(labels)*self.min_number_points)
            min_number_points = min_number_points if min_number_points >= 0 else 1
            
            features_data, labels, features_point_indices = filterFeaturesData(features_data, labels, types=self.filter_features_parameters['surface_types'],
                                                                               min_number_points=min_number_points, features_point_indices=features_point_indices)

        self.filenames_by_set[self.current_set_name].append(filename)

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

            if noise_limit == 0.:
                noisy_points = gt_points
            else:
                self.normalization_parameters['add_noise'] = noise_limit
                noisy_points, _, _, _ = normalize(points, self.normalization_parameters, normals=normals)
                
            h5_file.create_dataset('noisy_points', data=noisy_points)

            if labels is not None:
                h5_file.create_dataset('gt_labels', data=labels)

                for i, feature in enumerate(features_data):
                    if feature is not None:
                        soup_name = f'feature_{i}'
                        grp = h5_file.create_group(soup_name)
                        grp.create_dataset('indices', data=features_point_indices[i])
                        feature['name'] = soup_name
                        feature['normalized'] = True
                        if 'vert_indices' in feature.keys():
                            del feature['vert_indices']
                        if 'vert_parameters' in feature.keys():
                            del feature['vert_parameters']
                        if 'face_indices' in feature.keys():
                            del feature['face_indices']
                        sub_grp = grp.create_group('parameters')
                        dict_to_hdf5_group(sub_grp, feature)
                             
        return True

    def finish(self, permutation=None):
        train_models, val_models = self.divisionTrainVal(permutation=permutation)
        
        with open(os.path.join(self.data_folder_name, 'train_models.csv'), 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([f'{filename}.h5' for filename in train_models])
        with open(os.path.join(self.data_folder_name, 'test_models.csv'), 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([f'{filename}.h5' for filename in val_models])

        super().finish()