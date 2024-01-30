import pickle
from os.path import join
import os
import numpy as np
import statistics as stats

from .base_dataset_reader import BaseDatasetReader

from lib.normalization import applyTransforms, cubeRescale
from lib.utils import computeFeaturesPointIndices
from lib.fitting_func import FittingFunctions

class PrimitivenetDatasetReader(BaseDatasetReader):

    TYPES_MAP = {
        0: 'Plane',
        1: 'Cylinder',
        2: 'Cone',
        3: 'Sphere',
    }

    def __init__(self, parameters):
        super().__init__(parameters)

        train_path = os.path.join(self.data_folder_name, 'train')
        if os.path.exists(train_path):
            self.filenames_by_set['train'] = [f[:f.rfind('.')] for f in list(os.listdir(train_path))]
        else:
            self.filenames_by_set['train'] = []

        val_path = os.path.join(self.data_folder_name, 'val')
        if os.path.exists(val_path):
            self.filenames_by_set['val'] = [f[:f.rfind('.')] for f in list(os.listdir(val_path))]
        else:
            self.filenames_by_set['val'] = []

    def step(self):
        assert self.current_set_name in self.filenames_by_set.keys()

        index = self.steps_by_set[self.current_set_name]%len(self.filenames_by_set[self.current_set_name])
        filename = self.filenames_by_set[self.current_set_name][index]

        data_file_path = join(self.data_folder_name, self.current_set_name, f'{filename}.npz')
        transforms_file_path = join(self.transform_folder_name, f'{filename}.pkl')

        with open(transforms_file_path, 'rb') as pkl_file:
            transforms = pickle.load(pkl_file)

        with np.load(data_file_path, 'r') as npz_file:
            noisy_points = npz_file['V'] if 'V' in npz_file.keys() else None

            #TODO: fix this
            noisy_points = noisy_points - np.mean(noisy_points, axis=0)

            gt_points = npz_file['V_fixed'] if 'V' in npz_file.keys() else None
            noisy_normals = npz_file['N'] if 'N' in npz_file.keys() else None
            gt_normals = npz_file['N_fixed'] if 'N' in npz_file.keys() else None
            labels = npz_file['L'] if 'L' in npz_file.keys() else None
            semantics = npz_file['S'] if 'S' in npz_file.keys() else None 

            unique_labels = np.unique(labels)
            unique_labels = unique_labels[unique_labels != -1]

            if len(unique_labels) > 0:
                max_size = max(unique_labels) + 1
            else:
                max_size = 0

            fpi = computeFeaturesPointIndices(labels, size=max_size)

            points_scale = None
            features_data = [None]*max_size  
            for label in unique_labels:
                indices = fpi[label]
                types = semantics[indices]
                tp_id = stats.mode(types)

                feature = {}
                tp = PrimitivenetDatasetReader.TYPES_MAP[tp_id]
            
                points = noisy_points if not self.fit_noisy_points else gt_points
                normals = noisy_normals if not self.fit_noisy_normals else gt_normals

                if points_scale is None:
                    _, _, points_scale = cubeRescale(points.copy())

                feature = FittingFunctions.fit(tp, points[indices], normals[indices], scale=1/points_scale)
                features_data[label] = feature

        if self.unnormalize:
            gt_points, gt_normals, features_data = applyTransforms(gt_points, transforms, normals=gt_normals, features=features_data)
            noisy_points, noisy_normals, _ = applyTransforms(noisy_points, transforms, normals=noisy_normals, features=[])

        result = {
            'noisy_points': noisy_points,
            'points': gt_points,
            'noisy_normals': noisy_normals,
            'normals': gt_normals,
            'labels': labels,
            'features_data': features_data,
            'filename': filename,
        }

        self.steps_by_set[self.current_set_name] += 1
        
        return result
    
    def finish(self):
        super().finish()