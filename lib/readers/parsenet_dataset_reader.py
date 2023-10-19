import pickle
import h5py
from os.path import join, exists
import numpy as np

from .base_dataset_reader import BaseDatasetReader

from lib.normalization import applyTransforms
from lib.utils import computeFeaturesPointIndices
from lib.fitting_func import FittingFunctions

def get_data_at_index(data, index):
        partial_data = {}
        for key, value in data.items():
            if value is not None:
                if len(value.shape) == 2:
                    partial_data[key] = value[index, :]
                elif len(value.shape) == 3:
                    partial_data[key] = value[index, :, :]
        
        return partial_data

def collect_data_from_h5(h5_file):
    data = {}
    data['points'] = h5_file['points'][()] if 'points' in h5_file.keys() else None
    data['normals'] = h5_file['normals'][()] if 'normals' in h5_file.keys() else None
    data['labels'] = h5_file['labels'][()] if 'labels' in h5_file.keys() else None
    data['prim'] = h5_file['prim'][()] if 'prim' in h5_file.keys() else None
    data['gt_indices'] = h5_file['gt_indices'][()] if 'gt_indices' in h5_file.keys() else None
    data['matching'] = h5_file['matching'][()] if 'matching' in h5_file.keys() else None
    data['global_indices'] = h5_file['global_indices'][()] if 'global_indices' in h5_file.keys() else None
    return data

class ParsenetDatasetReader(BaseDatasetReader):
    PRIMITIVES_MAP = {
        1: 'Plane',
        3: 'Cone',
        4: 'Cylinder',
        5: 'Sphere' 
    }

    def read_data(self, set_name):
        self.data_by_set[set_name] = {}
        path = join(self.data_folder_name, f'{set_name}_data.h5')
        if exists(path):
            with h5py.File(path, 'r') as h5_file:
                data = collect_data_from_h5(h5_file)

        path = join(self.data_folder_name, f'{set_name}_ids.txt')
        with open(path, 'r') as txt_file:
            filenames = txt_file.read().split('\n')
            if len(filenames[-1].strip()) == 0:
               filenames.pop(-1) 

        for i, filename in enumerate(filenames):
            self.data_by_set[set_name][filename] = get_data_at_index(data, i) 

        self.filenames_by_set[set_name] = filenames

    def __init__(self, parameters):
        self.data_by_set = {}
        super().__init__(parameters)

        self.read_data('train')
        self.read_data('val')

    def step(self, unormalize=True, **kwargs):
        assert self.current_set_name in self.filenames_by_set.keys()

        index = self.steps_by_set[self.current_set_name]%len(self.filenames_by_set[self.current_set_name])
        filename = self.filenames_by_set[self.current_set_name][index]

        transforms_file_path = join(self.transform_folder_name, f'{filename}.pkl')

        with open(transforms_file_path, 'rb') as pkl_file:
            transforms = pickle.load(pkl_file)

        data = self.data_by_set[self.current_set_name][filename]
        
        points = data['points'] if 'points' in data.keys() else None
        normals = data['normals'] if 'normals' in data.keys() else None
        labels = data['labels'] if 'labels' in data.keys() else None
        prim = data['prim'] if 'prim' in data.keys() else None
        #params = h5_file['T_param'] if 'T_param' in h5_file.keys() else None
        #local_2_global_map = h5_file['local_2_global_map'] if 'local_2_global_map' in h5_file.keys() else None
        gt_indices = data['gt_indices'] if 'gt_indices' in data.keys() else None
        matching = data['matching'] if 'matching' in data.keys() else None
        global_indices = data['global_indices'] if 'global_indices' in data.keys() else None

        # if local_2_global_map is not None:
        #     valid_labels_mask = labels != -1
        #     labels[valid_labels_mask] = local_2_global_map[labels[valid_labels_mask]]

        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]

        if len(unique_labels) > 0:
            max_size = max(unique_labels) + 1
        else:
            max_size = 0

        fpi = computeFeaturesPointIndices(labels, size=max_size)

        features_data = [None]*max_size  
        for label in unique_labels:
            indices = fpi[label]
            types = prim[indices]
            types_unique, types_counts = np.unique(types, return_counts=True)
            argmax = np.argmax(types_counts)
            tp_id = types_unique[argmax]

            valid_indices = indices[np.where(types==tp_id)[0].astype(np.int32)]

            feature = {}
            tp = ParsenetDatasetReader.PRIMITIVES_MAP[tp_id]
            feature['type'] = tp

            primitive_params = FittingFunctions.fit(tp, points[indices], normals[indices])

            if tp == 'Plane':
                z_axis, d = primitive_params
                feature['z_axis'] = z_axis.tolist()
                feature['location'] = (d*z_axis).tolist()

            elif tp == 'Cone':
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
                z_axis, location, radius = primitive_params
                if radius > 10 or location[0] > 10 or location[1] > 10 or location[2] > 10:
                    radius = -1

                feature['z_axis'] = z_axis.tolist()
                feature['location'] = location.tolist()
                feature['radius'] = radius

            elif tp == 'Sphere':
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