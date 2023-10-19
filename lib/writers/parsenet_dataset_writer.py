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

    def fillH5File(h5_file, file_labels, points, normals, labels, 
                   primitives, gt_indices=None, matching=None, global_indices=None):
        if len(file_labels) == 0:
            return

        points_curr = points[file_labels]
        normals_curr = normals[file_labels]
        labels_curr = labels[file_labels]
        primitives_curr = primitives[file_labels]
        gt_indices_curr = gt_indices[file_labels] if gt_indices is not None else None
        matching_curr = matching[file_labels] if matching is not None else None
        global_indices_curr = global_indices[file_labels] if global_indices is not None else None

        h5_file.create_dataset('points', data=points_curr)
        h5_file.create_dataset('normals', data=normals_curr)
        h5_file.create_dataset('labels', data=labels_curr)
        h5_file.create_dataset('prim', data=primitives_curr)
        if gt_indices_curr is not None:
            h5_file.create_dataset('gt_indices', data=gt_indices_curr)
        if matching_curr is not None:
            h5_file.create_dataset('matching', data=matching_curr)
        if global_indices_curr is not None:
            h5_file.create_dataset('global_indices', data=global_indices_curr)

    def __init__(self, parameters):
        super().__init__(parameters)

        self.names = {}
        self.points = []
        self.normals = []
        self.labels = []
        self.primitives = []
        self.gt_indices = []
        self.matching = []
        self.global_indices = []

    def step(self, points, normals=None, labels=None, features_data=[], noisy_points=None,
             noisy_normals=None, filename=None, features_point_indices=None, **kwargs):
        
        if filename is None:
            filename = str(uuid.uuid4())

        lengths = [len(p) for p in self.points]

        if len(lengths) > 0 and len(points) not in lengths:
            print(f'ERROR: all models must have the same number of points. Model {filename} is not going to be writed.')
            return False

        transforms_file_path = os.path.join(self.transform_folder_name, f'{filename}.pkl')

        if labels is not None:   
            if features_point_indices is None:
                features_point_indices = computeFeaturesPointIndices(labels, size=len(features_data))

            min_number_points = self.min_number_points if self.min_number_points >= 1 else int(len(labels)*self.min_number_points)
            min_number_points = min_number_points if min_number_points >= 0 else 1
            
            features_data, labels, features_point_indices = filterFeaturesData(features_data, labels, types=self.surface_types,
                                                                               min_number_points=min_number_points, 
                                                                               features_point_indices=features_point_indices)
            if len(features_data) == 0:
                print(f'WARNING: {filename} has no features left.')
    
        points, noisy_points, normals, noisy_normals, features_data, transforms = self.normalize(points, noisy_points, normals,
                                                                                        noisy_normals, features_data)
        with open(transforms_file_path, 'wb') as pkl_file:
            pickle.dump(transforms, pkl_file)

        if 'gt_indices' in kwargs:
            self.gt_indices.append(kwargs['gt_indices'])
        if 'matching' in kwargs:
            self.matching.append(kwargs['matching'])
        if 'global_indices' in kwargs:
            self.global_indices.append(kwargs['global_indices'])
        
        self.filenames_by_set[self.current_set_name].append(filename)

        self.names[filename] = len(self.points)
        self.points.append(points)
        if normals is not None:
            self.normals.append(normals)
        self.labels.append(labels)
        primitives = [ParsenetDatasetWriter.FEATURES_ID[features_data[lab]['type'].lower()] if lab != -1 else -1 for lab in labels]
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

        points = np.asarray(self.points, dtype=np.float64)
        normals = np.asarray(self.normals, dtype=np.float64)
        labels = np.asarray(self.labels, dtype=np.int32)
        primitives = np.asarray(self.primitives, dtype=np.int32)
        gt_indices = np.asarray(self.gt_indices, dtype=np.int32) if len(self.gt_indices) == len(points) else None
        matching = np.asarray(self.matching, dtype=np.int32) if len(self.matching) == len(points) else None
        global_indices = np.asarray(self.global_indices, dtype=np.int32) if len(self.global_indices) == len(points) else None

        if len(train_labels) > 0:
            with open(os.path.join(self.data_folder_name, 'train_ids.txt'), 'w') as f:
                f.write('\n'.join(train_models))
            with h5py.File(os.path.join(self.data_folder_name, 'train_data.h5'), 'w') as h5_file:
                ParsenetDatasetWriter.fillH5File(h5_file, train_labels, points, normals, labels, primitives, 
                                                 gt_indices=gt_indices, matching=matching, global_indices=global_indices)

        if len(val_labels) > 0:
            with open(os.path.join(self.data_folder_name, 'val_ids.txt'), 'w') as f:
                f.write('\n'.join(val_models))
            with h5py.File(os.path.join(self.data_folder_name, 'val_data.h5'), 'w') as h5_file:
                ParsenetDatasetWriter.fillH5File(h5_file, val_labels, points, normals, labels, primitives,
                                                 gt_indices=gt_indices, matching=matching, global_indices=global_indices)

        super().finish()