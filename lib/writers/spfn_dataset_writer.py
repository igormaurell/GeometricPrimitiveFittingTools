import pickle
import h5py
import csv
import numpy as np
import gc
import uuid
import os

from lib.normalization import normalize
from lib.utils import filterFeaturesData, translateFeature, computeLabelsFromFace2Primitive, computeFeaturesPointIndices, strLower

from .base_dataset_writer import BaseDatasetWriter

class SpfnDatasetWriter(BaseDatasetWriter):
    FEATURES_BY_TYPE = {
        'plane': ['type', 'name', 'location_x', 'location_y', 'location_z', 'axis_x', 'axis_y', 'axis_z', 'normalized'],
        'cylinder': ['type', 'name', 'location_x', 'location_y', 'location_z', 'axis_x', 'axis_y', 'axis_z', 'radius', 'normalized'],
        'cone': ['type', 'name', 'location_x', 'location_y', 'location_z', 'axis_x', 'axis_y', 'axis_z', 'radius', 'semi_angle', 'apex_x', 'apex_y', 'apex_z', 'normalized'],
        'sphere': ['type', 'name', 'location_x', 'location_y', 'location_z', 'radius', 'normalized']
    }

    FEATURES_MAPPING = {
        'type': {'type': str, 'map': 'type', 'transform': strLower},
        'name': {'type': str, 'map': 'name'},
        'normalized': {'type': bool, 'map': 'normalized'},
        'location_x': {'type': float, 'map': ('location', 0)},
        'location_y': {'type': float, 'map': ('location', 1)},
        'location_z': {'type': float, 'map': ('location', 2)},
        'axis_x': {'type': float, 'map': ('z_axis', 0)},
        'axis_y': {'type': float, 'map': ('z_axis', 1)},
        'axis_z': {'type': float, 'map': ('z_axis', 2)},
        'apex_x': {'type': float, 'map': ('apex', 0)},
        'apex_y': {'type': float, 'map': ('apex', 1)},
        'apex_z': {'type': float, 'map': ('apex', 2)},
        'semi_angle': {'type': float, 'map': 'angle'},
        'radius': {'type': float, 'map': 'radius'},
    }

    def __init__(self, parameters):
        super().__init__(parameters)

    def step(self, points, normals=None, labels=None, features_data=[], noisy_points=None, filename=None, features_point_indices=None):
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
                features_point_indices = computeFeaturesPointIndices(labels)

            min_number_points = self.min_number_points if self.min_number_points > 1 else int(len(labels)*self.min_number_points)
            min_number_points = min_number_points if min_number_points >= 0 else 1

            features_data, labels, features_point_indices = filterFeaturesData(features_data, types=self.filter_features_parameters['surface_types'], min_number_points=min_number_points,
                                                           labels=labels, features_point_indices=features_point_indices)

            if len(features_data) == 0:
                print(f'ERROR: {data_file_path} has no features left.')
                return False

        self.filenames_by_set[self.current_set_name].append(filename)

        with h5py.File(data_file_path, 'w') as h5_file:
            noise_limit = 0.

            if 'add_noise' in self.normalization_parameters.keys():
                noise_limit = self.normalization_parameters['add_noise']
                self.normalization_parameters['add_noise'] = 0.
                
            points, gt_normals, features_data, transforms = normalize(points, self.normalization_parameters, normals=normals.copy(), features=features_data)

            with open(transforms_file_path, 'wb') as pkl_file:
                pickle.dump(transforms, pkl_file)

            h5_file.create_dataset('gt_points', data=points)
            if gt_normals is not None:
                h5_file.create_dataset('gt_normals', data=gt_normals)

            #if noise_limit != 0. or noisy_points is not None:
            if noisy_points is None:
                noisy_points = points.copy()
            self.normalization_parameters['add_noise'] = noise_limit
            noisy_points, _, _, _ = normalize(noisy_points, self.normalization_parameters, normals=normals.copy())
            h5_file.create_dataset('noisy_points', data=noisy_points)
            del noisy_points

            del gt_normals
            gc.collect()

            if labels is not None:
                h5_file.create_dataset('gt_labels', data=labels)

                point_position = data_file_path.rfind('.')
                point_position = point_position if point_position >= 0 else len(point_position)
                bar_position = data_file_path.rfind('/')
                bar_position = bar_position if bar_position >= 0 else 0

                for i, feature in enumerate(features_data):
                    soup_name = f'{filename}_soup_{i}'
                    grp = h5_file.create_group(soup_name)
                    feat_points = points[features_point_indices[i]]
                    grp.create_dataset('gt_points', data=feat_points)
                    feature['name'] = soup_name
                    feature['normalized'] = True
                    feature = translateFeature(feature, SpfnDatasetWriter.FEATURES_BY_TYPE, SpfnDatasetWriter.FEATURES_MAPPING)
                    grp.attrs['meta'] = np.void(pickle.dumps(feature))
                             
        return True

    def finish(self, permutation=None):
        train_models, test_models = self.divisionTrainVal(permutation=permutation)
        
        with open(os.path.join(self.data_folder_name, 'train_models.csv'), 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([f'{filename}.h5' for filename in train_models])
        with open(os.path.join(self.data_folder_name, 'test_models.csv'), 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([f'{filename}.h5' for filename in test_models])

        super().finish()