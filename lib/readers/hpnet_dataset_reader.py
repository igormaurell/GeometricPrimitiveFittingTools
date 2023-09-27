import pickle
import h5py
from os.path import join
import numpy as np

from .base_dataset_reader import BaseDatasetReader

from lib.normalization import applyTransforms
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

    def step(self, unormalize=True, **kwargs):
        assert self.current_set_name in self.filenames_by_set.keys()

        index = self.steps_by_set[self.current_set_name]%len(self.filenames_by_set[self.current_set_name])
        filename = self.filenames_by_set[self.current_set_name][index]

        data_file_path = join(self.data_folder_name, f'{filename}.h5')
        transforms_file_path = join(self.transform_folder_name, f'{filename}.pkl')

        with open(transforms_file_path, 'rb') as pkl_file:
            transforms = pickle.load(pkl_file)
        
        with h5py.File(data_file_path, 'r') as h5_file:
            points = h5_file['points'][()] if 'points' in h5_file.keys() else None
            normals = h5_file['normals'][()] if 'normals' in h5_file.keys() else None
            labels = h5_file['labels'][()] if 'labels' in h5_file.keys() else None
            prim = h5_file['prim'][()] if 'prim' in h5_file.keys() else None
            params = h5_file['T_param'][()] if 'T_param' in h5_file.keys() else None
            local_2_global_map = h5_file['local_2_global_map'][()] if 'local_2_global_map' in h5_file.keys() else None
            gt_indices = h5_file['gt_indices'][()] if 'gt_indices' in h5_file.keys() else None
            matching = h5_file['matching'][()] if 'matching' in h5_file.keys() else None
            global_indices = h5_file['global_indices'][()] if 'global_indices' in h5_file.keys() else None

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

            features_data = [None]*max_size  
            for label in unique_labels:
                indices = fpi[label]
                types = prim[indices]
                types_unique, types_counts = np.unique(types, return_counts=True)
                argmax = np.argmax(types_counts)
                tp_id = types_unique[argmax]

                valid_indices = indices[np.where(types==tp_id)[0].astype(np.int32)]

                feature = {}
                tp = HPNetDatasetReader.PRIMITIVES_MAP[tp_id]
                feature['type'] = tp

                if use_data_primitives:
                    primitive_params = params[valid_indices, :]
                else:
                    primitive_params = FittingFunctions.fit(tp, points[indices], normals[indices])

                if tp == 'Plane':
                    if use_data_primitives:
                        z_axis, d = primitive_params[0, 4:7], primitive_params[0, 7]
                    else:
                        z_axis, d = primitive_params
                    feature['z_axis'] = z_axis.tolist()
                    feature['location'] = (d*z_axis).tolist()

                elif tp == 'Cone':
                    if use_data_primitives:
                        location, apex, angle = primitive_params[0, 18:21], primitive_params[0, 15:18], primitive_params[0, 21]
                    else:
                        location, apex, angle = primitive_params

                    axis = location - apex
                    dist = np.linalg.norm(axis)
                    z_axis = axis/dist
                    radius = np.tan(angle)*dist

                    feature['angle'] = angle
                    feature['apex'] = apex.tolist()
                    feature['location'] = location.tolist()
                    feature['z_axis'] = z_axis.tolist()
                    feature['radius'] = radius

                elif tp == 'Cylinder':
                    if use_data_primitives:
                        z_axis, location, radius = primitive_params[0, 8:11], primitive_params[0, 11:14], primitive_params[0, 14]
                    else:
                        z_axis, location, radius = primitive_params
                        if radius > 10 or location[0] > 10 or location[1] > 10 or location[2] > 10:
                            radius = -1

                    feature['z_axis'] = z_axis.tolist()
                    feature['location'] = location.tolist()
                    feature['radius'] = radius

                elif tp == 'Sphere':
                    if use_data_primitives:
                        location, radius = primitive_params[0, :3], primitive_params[0, 3]
                    else:
                        location, radius = primitive_params
                        
                    feature['location'] = location.tolist()
                    feature['radius'] = radius
                
                features_data[label] = feature

            if unormalize:
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