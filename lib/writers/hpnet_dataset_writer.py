import pickle
import h5py
import csv
import numpy as np
import gc
import uuid
import os
from collections.abc import Iterable

from lib.normalization import normalize
from lib.utils import filterFeaturesData, computeFeaturesPointIndices

from .base_dataset_writer import BaseDatasetWriter

class HPNetDatasetWriter(BaseDatasetWriter):
    PRIMITIVES_MAP = {
        'Plane': 1,
        'Cone': 3,
        'Cylinder': 4,
        'Sphere': 5
    }

    def __init__(self, parameters):
        super().__init__(parameters)

    def step(self, points, normals=None, labels=None, features_data=[], noisy_points=None, filename=None, features_point_indices=None, **kwargs):

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
            
            features_data, labels, features_point_indices = filterFeaturesData(features_data, labels, types=self.surface_types,
                                                                               min_number_points=min_number_points, 
                                                                               features_point_indices=features_point_indices)
            if len(features_data) == 0:
                print(f'WARNING: {data_file_path} has no features left.')

        self.filenames_by_set[self.current_set_name].append(filename)

        with h5py.File(data_file_path, 'w') as h5_file:
                
            points, normals, features_data, transforms = normalize(points.copy(), self.normalization_parameters, normals=normals.copy(), features=features_data)

            with open(transforms_file_path, 'wb') as pkl_file:
                pickle.dump(transforms, pkl_file)
                
            h5_file.create_dataset('points', data=points)
            h5_file.create_dataset('normals', data=normals)

            if labels is not None:
                primitive_params = np.zeros((len(labels), 22))
                types = np.zeros(len(labels)) - 1
                global_to_local_labels_map = np.zeros(len(features_data)) - 1
                j = 0
                for i, feature in enumerate(features_data):
                    if feature is not None and feature['type'] in HPNetDatasetWriter.PRIMITIVES_MAP:
                        tp = feature['type']
                        global_to_local_labels_map[i] = j
                        indices = features_point_indices[i]

                        types[indices] = HPNetDatasetWriter.PRIMITIVES_MAP[tp]

                        if tp == 'Plane':
                            z_axis = np.asarray(feature['z_axis'])
                            if 'foward' in feature and not feature['foward']:
                                z_axis = -z_axis
                            location = np.asarray(feature['location'])
                            primitive_params[indices, 4:7] = z_axis
                            primitive_params[indices, 7] = np.dot(location, z_axis)
                        elif tp == 'Cone':
                            primitive_params[indices, 15:18] = np.asarray(feature['apex'])
                            primitive_params[indices, 18:21] = np.asarray(feature['location'])
                            primitive_params[indices, 21] = feature['angle']
                        elif tp == 'Cylinder':
                            primitive_params[indices, 8:11] = np.asarray(feature['z_axis'])
                            primitive_params[indices, 11:14] = np.asarray(feature['location'])
                            primitive_params[indices, 14] = feature['radius']
                        elif tp == 'Sphere':
                            primitive_params[indices, :3] = np.asarray(feature['location'])
                            primitive_params[indices, 3] = feature['radius']

                        j+= 1

                local_labels = global_to_local_labels_map[labels]
                labels[local_labels == -1] = -1

                h5_file.create_dataset('prim', data=types.astype(np.int32))
                h5_file.create_dataset('T_param',  data=primitive_params)
                h5_file.create_dataset('labels', data=local_labels.astype(np.int32))
                h5_file.create_dataset('global_labels', data=labels.astype(np.int32))
                
        return True

    def finish(self, permutation=None):
        train_models, val_models = self.divisionTrainVal(permutation=permutation)
        
        with open(os.path.join(self.data_folder_name, 'train_data.txt'), 'w') as f:
            f.write('\n'.join(train_models))
        with open(os.path.join(self.data_folder_name, 'val_data.txt'), 'w') as f:
            f.write('\n'.join(val_models))
        with open(os.path.join(self.data_folder_name, 'test_data.txt'), 'w') as f:
            f.write('\n'.join(val_models))

        super().finish()