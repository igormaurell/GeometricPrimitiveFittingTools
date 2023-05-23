import pickle
import h5py
import numpy as np
import uuid
import os

from collections.abc import Iterable

from lib.normalization import normalize
from lib.utils import filterFeaturesData, computeFeaturesPointIndices

from operator import itemgetter

from .base_dataset_writer import BaseDatasetWriter

class ParsenetDatasetWriter(BaseDatasetWriter):
    FEATURES_ID = {
        'plane': 1,
        'cone': 3,
        'cylinder': 4,
        'sphere': 5
    }

    def fillH5File(h5_file, file_labels, points, normals, labels, primitives):
        if len(file_labels) == 0:
            return

        points_curr = points[file_labels]
        normals_curr = normals[file_labels]
        labels_curr = labels[file_labels]
        primitives_curr = primitives[file_labels]

        h5_file.create_dataset('points', data=points_curr)
        h5_file.create_dataset('normals', data=normals_curr)
        h5_file.create_dataset('labels', data=labels_curr)
        h5_file.create_dataset('prim', data=primitives_curr)

    def __init__(self, parameters):
        super().__init__(parameters)

        self.names = {}
        self.points = []
        self.normals = []
        self.labels = []
        self.primitives = []

        self.features_types = list(set(ParsenetDatasetWriter.FEATURES_ID.keys()) & set(self.filter_features_parameters['surface_types']))

    def step(self, points, normals=None, labels=None, features_data=[], noisy_points=None, filename=None, features_point_indices=None):
        if filename is None:
            filename = str(uuid.uuid4())
        
        if type(features_data) == dict:
            features_data = features_data['surfaces']

        transforms_file_path = os.path.join(self.transform_folder_name, f'{filename}.pkl')

        if labels is not None:
            if features_point_indices is None:
                features_point_indices = computeFeaturesPointIndices(labels, size=len(features_data))

            min_number_points = self.min_number_points if self.min_number_points >= 1 else int(len(labels)*self.min_number_points)
            min_number_points = min_number_points if min_number_points >= 0 else 1


            features_data, labels, features_point_indices = filterFeaturesData(features_data, types=self.features_types, min_number_points=min_number_points,
                                                           labels=labels, features_point_indices=features_point_indices)

            if len(features_data) == 0:
                #print(f'ERROR: {filename} has no features left.')
                return False
        else:
            return False
    
        points, normals, features_data, transforms = normalize(points, self.normalization_parameters, normals=normals.copy(), features=features_data)

        with open(transforms_file_path, 'wb') as pkl_file:
            pickle.dump(transforms, pkl_file)

        self.filenames_by_set[self.current_set_name].append(filename)

        self.names[filename] = len(self.points)
        self.points.append(points)
        if normals is not None:
            self.normals.append(normals)
        self.labels.append(labels)
        primitives = [ParsenetDatasetWriter.FEATURES_ID[features_data[labels[i]]['type'].lower()] for i in range(len(labels))]
        self.primitives.append(primitives)

        return True

    def finish(self, permutation=None):
        train_models, val_models = self.divisionTrainVal(permutation=permutation)
        
        tl = itemgetter(*train_models)(self.names) if len(train_models) > 0 else []
        tl = tl if isinstance(tl, Iterable) else [tl]
        train_labels = np.array(tl, dtype=np.int64)
        tl = itemgetter(*val_models)(self.names) if len(val_models) > 0 else []
        tl = tl if isinstance(tl, Iterable) else [tl]
        val_labels = np.array(tl, dtype=np.int64)

        points = np.array(self.points)
        normals = np.array(self.normals)
        labels = np.array(self.labels)
        primitives = np.array(self.primitives)

        if len(train_labels) > 0:
            with open(os.path.join(self.data_folder_name, 'train_ids.txt'), 'w') as f:
                f.write('\n'.join(train_models))
            with h5py.File(os.path.join(self.data_folder_name, 'train_data.h5'), 'w') as h5_file:
                ParsenetDatasetWriter.fillH5File(h5_file, train_labels, points, normals, labels, primitives)

        if len(val_labels) > 0:
            with open(os.path.join(self.data_folder_name, 'val_ids.txt'), 'w') as f:
                f.write('\n'.join(val_models))
            with h5py.File(os.path.join(self.data_folder_name, 'val_data.h5'), 'w') as h5_file:
                ParsenetDatasetWriter.fillH5File(h5_file, val_labels, points, normals, labels, primitives)

        super().finish()