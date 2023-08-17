import pickle
import h5py
import csv
from os.path import join
import os
import re
import numpy as np

from .base_dataset_reader import BaseDatasetReader

from lib.normalization import unNormalize
from lib.utils import translateFeature, strUpperFirstLetter

class PrimitivenetDatasetReader(BaseDatasetReader):
    FEATURES_BY_TYPE = {
        'plane': ['type', 'location', 'z_axis'],
        'cylinder': ['type', 'location', 'z_axis', 'radius'],
        'cone': ['type', 'location', 'z_axis', 'radius', 'angle', 'apex'],
        'sphere': ['type', 'location', 'radius']
    }

    FEATURES_MAPPING = {
        'type': {'type': str, 'map': 'type', 'transform': strUpperFirstLetter},
        'location': {'type': list, 'map': ['location_x', 'location_y', 'location_z']},
        'z_axis': {'type': list, 'map': ['axis_x', 'axis_y', 'axis_z']},
        'apex': {'type': list, 'map': ['apex_x', 'apex_y', 'apex_z']},
        'angle': {'type': float, 'map': 'semi_angle'},
        'radius': {'type': float, 'map': 'radius'},
    }

    def __init__(self, parameters):
        super().__init__(parameters)

        self.filenames_by_set['train'] = [os.path.join('train', filename) for filename in list(os.listdir(os.path.join(self.data_folder_name, 'train')))]
        self.filenames_by_set['val'] = [os.path.join('val', filename) for filename in list(os.listdir(os.path.join(self.data_folder_name, 'val')))]

    def step(self):
        assert self.current_set_name in self.filenames_by_set.keys()

        index = self.steps_by_set[self.current_set_name]%len(self.filenames_by_set[self.current_set_name])
        filename = self.filenames_by_set[self.current_set_name][index]
        point_position = filename.rfind('.')

        data_file_path = join(self.data_folder_name, filename)
        filename = filename[:point_position]
        transforms_file_path = join(self.transform_folder_name, f'{filename}.pkl')

        with open(transforms_file_path, 'rb') as pkl_file:
            transforms = pickle.load(pkl_file)

        with np.load(data_file_path, 'r') as npz_file:
            noisy_points = npz_file['noisy_points'] if 'noisy_points' in npz_file.keys() else None
            gt_points = npz_file['V'] if 'V' in npz_file.keys() else None
            gt_normals = npz_file['N'] if 'N' in npz_file.keys() else None
            labels = npz_file['I'] if 'I' in npz_file.keys() else None

            found_soup_ids = []
            soup_id_to_key = {}
            soup_prog = re.compile('(.*)_soup_([0-9]+)$')
            for key in list(npz_file.keys()):
                m = soup_prog.match(key)
                if m is not None:
                    soup_id = int(m.group(2))
                    found_soup_ids.append(soup_id)
                    soup_id_to_key[soup_id] = key

            features_data = []            
            found_soup_ids.sort()
            for i in range(len(found_soup_ids)):
                g = npz_file[soup_id_to_key[i]]
                meta = pickle.loads(g.attrs['meta'])
                meta = translateFeature(meta, PrimitivenetDatasetReader.FEATURES_BY_TYPE, PrimitivenetDatasetReader.FEATURES_MAPPING)
                features_data.append(meta)

        gt_points, gt_normals, features_data = unNormalize(gt_points, transforms, normals=gt_normals, features=features_data)
        noisy_points, _, _ = unNormalize(noisy_points, transforms, normals=None, features=[])

        result = {
            'noisy_points': noisy_points,
            'points': gt_points,
            'normals': gt_normals,
            'labels': labels,
            'features_data': features_data,
            'filename': filename,
        }

        self.steps_by_set[self.current_set_name] += 1
        
        return result
    
    def finish(self):
        super().finish()