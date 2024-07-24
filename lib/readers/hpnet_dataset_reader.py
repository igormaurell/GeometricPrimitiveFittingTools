import pickle
import h5py
from os.path import join
import numpy as np
import statistics as stats

from .base_dataset_reader import BaseDatasetReader

from lib.normalization import applyTransforms, cubeRescale
from lib.utils import computeFeaturesPointIndices
from lib.fitting_func import FittingFunctions

class HPNetDatasetReader(BaseDatasetReader):
    PRIMITIVES_MAP = {
        1: 'Plane',
        3: 'Cone',
        4: 'Cylinder',
        5: 'Sphere' 
    }

    def __init__(self, parameters):
        super().__init__(parameters)

        with open(join(self.data_folder_name, 'train_data.txt'), 'r') as f:
            read = f.read()
            self.filenames_by_set['train'] = read.split('\n')
            if self.filenames_by_set['train'][-1] == '':
                self.filenames_by_set['train'].pop()
        with open(join(self.data_folder_name, 'val_data.txt'), 'r') as f:
            read = f.read()
            self.filenames_by_set['val'] = read.split('\n')
            if self.filenames_by_set['val'][-1] == '':
                self.filenames_by_set['val'].pop()

    def step(self, **kwargs):
        assert self.current_set_name in self.filenames_by_set.keys()

        index = self.steps_by_set[self.current_set_name]%len(self.filenames_by_set[self.current_set_name])
        filename = self.filenames_by_set[self.current_set_name][index]

        data_file_path = join(self.data_folder_name, f'{filename}.h5')
        transforms_file_path = join(self.transform_folder_name, f'{filename}.pkl')

        with open(transforms_file_path, 'rb') as pkl_file:
            transforms = pickle.load(pkl_file)
        
        with h5py.File(data_file_path, 'r') as h5_file:
            points = h5_file['points'][()].astype(np.float32) if 'points' in h5_file.keys() else None
            normals = h5_file['normals'][()].astype(np.float32) if 'normals' in h5_file.keys() else None
            labels = h5_file['labels'][()].astype(np.int32) if 'labels' in h5_file.keys() else None
            prim = h5_file['prim'][()].astype(np.int32) if 'prim' in h5_file.keys() else None
            params = h5_file['T_param'][()].astype(np.float32) if 'T_param' in h5_file.keys() else None
            local_2_global_map = h5_file['local_2_global_map'][()].astype(np.int32) if 'local_2_global_map' in h5_file.keys() else None
            gt_indices = h5_file['gt_indices'][()].astype(np.int32) if 'gt_indices' in h5_file.keys() else None
            matching = h5_file['matching'][()].astype(np.int32) if 'matching' in h5_file.keys() else None
            global_indices = h5_file['global_indices'][()].astype(np.int32) if 'global_indices' in h5_file.keys() else None

            if local_2_global_map is not None:
                valid_labels_mask = labels != -1
                labels[valid_labels_mask] = local_2_global_map[labels[valid_labels_mask]]

            unique_labels = np.unique(labels)
            unique_labels = unique_labels[unique_labels != -1]

            if len(unique_labels) > 0:
                max_size = max(unique_labels) + 1
            else:
                max_size = 0

            fpi = computeFeaturesPointIndices(labels, size=max_size)

            use_data_primitives = self.use_data_primitives or params is None

            points_scale = None
            features_data = [None]*max_size  
            for label in unique_labels:
                indices = fpi[label]
                types = prim[indices]
                tp_id = stats.mode(types)

                valid_indices = indices[np.where(types==tp_id)[0].astype(np.int32)]

                feature = {}
                tp = HPNetDatasetReader.PRIMITIVES_MAP[tp_id]

                if use_data_primitives:
                    primitive_params = params[valid_indices, :]
                    params_curr = None
                    if tp == 'Plane':
                        params_curr = (primitive_params[0, 4:7], primitive_params[0, 7])
                    elif tp == 'Cone':
                        params_curr = (primitive_params[0, 18:21], primitive_params[0, 15:18], primitive_params[0, 21])
                    elif tp == 'Cylinder':
                        params_curr = (primitive_params[0, 8:11], primitive_params[0, 11:14], primitive_params[0, 14])
                    elif tp == 'Sphere':
                        params_curr = (primitive_params[0, :3], primitive_params[0, 3])
                    feature = FittingFunctions.params2dict(params_curr, tp)
                else:
                    if points_scale is None:
                        _, _, points_scale = cubeRescale(points.copy())
                    feature = FittingFunctions.fit(tp, points[indices], normals[indices], scale=1/points_scale)
                
                features_data[label] = feature

            if self.unnormalize:
                points, normals, features_data = applyTransforms(points, transforms, normals=normals, features=features_data)
            

        result = {
            'noisy_points': points.copy(),
            'points': points,
            'noisy_normals': normals.copy(),
            'normals': normals,
            'labels': labels,
            'features_data': features_data,
            'filename': filename,
            'transforms': transforms,
        }
        if gt_indices is not None:
            result['gt_indices'] = gt_indices
        if matching is not None:
            result['matching'] = matching
        if global_indices is not None:
            result['global_indices'] = global_indices

        self.steps_by_set[self.current_set_name] += 1
        
        return result
    
    def finish(self):
        super().finish()
    
    def __iter__(self):
        return super().__iter__()