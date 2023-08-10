import pickle
import h5py
from os.path import join
import numpy as np

from .base_dataset_reader import BaseDatasetReader

from lib.normalization import unNormalize

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

    def step(self, unormalize=True):
        assert self.current_set_name in self.filenames_by_set.keys()

        index = self.steps_by_set[self.current_set_name]%len(self.filenames_by_set[self.current_set_name])
        filename = self.filenames_by_set[self.current_set_name][index]

        data_file_path = join(self.data_folder_name, f'{filename}.h5')
        transforms_file_path = join(self.transform_folder_name, f'{filename}.pkl')

        with open(transforms_file_path, 'rb') as pkl_file:
            transforms = pickle.load(pkl_file)

        with h5py.File(data_file_path, 'r') as h5_file:
            print(h5_file.keys())
            points = h5_file['points'][()] if 'points' in h5_file.keys() else None
            normals = h5_file['normals'][()] if 'normals' in h5_file.keys() else None
            labels = h5_file['labels'][()] if 'labels' in h5_file.keys() else None
            prim = h5_file['prim'][()] if 'prim' in h5_file.keys() else None
            params = h5_file['T_param'][()] if 'T_param' in h5_file.keys() else None
            global_labels = h5_file['global_labels'][()] if 'global_labels' in h5_file.keys() else None
            gt_indices = h5_file['gt_indices'][()] if 'gt_indices' in h5_file.keys() else None
            matching = h5_file['matching'][()] if 'matching' in h5_file.keys() else None

            if global_labels is not None:
                unique_labels, unique_indices = np.unique(global_labels, return_index=True)
            else:
                unique_labels, unique_indices = np.unique(labels, return_index=True)

            unique_indices = unique_indices[unique_labels != -1]
            unique_labels = unique_labels[unique_labels != -1]
            
            types = prim[unique_indices]
            primitive_params = params[unique_indices]

            if len(unique_labels) > 0:
                max_size = max(unique_labels) + 1
            else:
                max_size = 0
            features_data = [None]*max_size  
            for i, label in enumerate(unique_labels):
                feature = {}
                tp = HPNetDatasetReader.PRIMITIVES_MAP[int(types[i])]
                feature['type'] = tp

                if tp == 'Plane':
                    z_axis = primitive_params[i, 4:7]
                    feature['z_axis'] = z_axis.tolist()
                    feature['location'] = (primitive_params[i, 7]*z_axis).tolist()

                elif tp == 'Cone':
                    apex = primitive_params[i, 15:18]
                    location = primitive_params[i, 18:21]
                    angle = primitive_params[i, 21]
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
                    feature['z_axis'] = primitive_params[i, 8:11].tolist()
                    feature['location'] = primitive_params[i, 11:14].tolist()
                    feature['radius'] = primitive_params[i, 14]

                elif tp == 'Sphere':
                    feature['location'] = primitive_params[i, :3].tolist()
                    feature['radius'] = primitive_params[i, 3]
                
                features_data[label] = feature

            if unormalize:
                points, normals, features_data = unNormalize(points, transforms, normals=normals, features=features_data)

        result = {
            'points': points,
            'normals': normals,
            'labels': global_labels if global_labels is not None else labels,
            'features': features_data,
            'filename': filename,
            'transforms': transforms,
        }
        if gt_indices is not None:
            result['gt_indices'] = gt_indices
        if matching is not None:
            result['matching'] = matching

        self.steps_by_set[self.current_set_name] += 1
        
        return result
    
    def finish(self):
        super().finish()
    
    def __iter__(self):
        return super().__iter__()