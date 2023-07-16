import pickle
import h5py
import csv
from os.path import join
import re

from .base_dataset_reader import BaseDatasetReader

from lib.normalization import unNormalize
from lib.utils import translateFeature, strUpperFirstLetter

class SpfnDatasetReader(BaseDatasetReader):
    FEATURES_BY_TYPE = {
        'plane': ['type', 'foward', 'location', 'z_axis'],
        'cylinder': ['type', 'foward', 'location', 'z_axis', 'radius'],
        'cone': ['type', 'foward', 'location', 'z_axis', 'radius', 'angle', 'apex'],
        'sphere': ['type', 'foward', 'location', 'radius']
    }

    FEATURES_MAPPING = {
        'type': {'type': str, 'map': 'type', 'transform': strUpperFirstLetter},
        'location': {'type': list, 'map': ['location_x', 'location_y', 'location_z']},
        'z_axis': {'type': list, 'map': ['axis_x', 'axis_y', 'axis_z']},
        'apex': {'type': list, 'map': ['apex_x', 'apex_y', 'apex_z']},
        'angle': {'type': float, 'map': 'semi_angle'},
        'radius': {'type': float, 'map': 'radius'},
        'foward': {'type': str, 'map': 'foward'}
    }

    def __init__(self, parameters):
        super().__init__(parameters)

        with open(join(self.data_folder_name, 'train_models.csv'), 'r', newline='') as f:
            self.filenames_by_set['train'] = list(csv.reader(f, delimiter=',', quotechar='|'))[0]
        with open(join(self.data_folder_name, 'test_models.csv'), 'r', newline='') as f:
            self.filenames_by_set['val'] = list(csv.reader(f, delimiter=',', quotechar='|'))[0]

    def step(self, unormalize=True):
        assert self.current_set_name in self.filenames_by_set.keys()

        index = self.steps_by_set[self.current_set_name]%len(self.filenames_by_set[self.current_set_name])
        filename = self.filenames_by_set[self.current_set_name][index]
        point_position = filename.rfind('.')

        data_file_path = join(self.data_folder_name, filename)
        filename = filename[:point_position]
        transforms_file_path = join(self.transform_folder_name, f'{filename}.pkl')

        with open(transforms_file_path, 'rb') as pkl_file:
            transforms = pickle.load(pkl_file)

        with h5py.File(data_file_path, 'r') as h5_file:
            noisy_points = h5_file['noisy_points'][()] if 'noisy_points' in h5_file.keys() else None
            gt_points = h5_file['gt_points'][()] if 'gt_points' in h5_file.keys() else None
            gt_normals = h5_file['gt_normals'][()] if 'gt_normals' in h5_file.keys() else None
            labels = h5_file['gt_labels'][()] if 'gt_labels' in h5_file.keys() else None

            found_soup_ids = []
            soup_id_to_key = {}
            soup_prog = re.compile('(.*)_soup_([0-9]+)$')
            for key in list(h5_file.keys()):
                m = soup_prog.match(key)
                if m is not None:
                    soup_id = int(m.group(2))
                    found_soup_ids.append(soup_id)
                    soup_id_to_key[soup_id] = key

            features_data = [None]*(max(found_soup_ids) + 1)      
            found_soup_ids.sort()
            for i in found_soup_ids:
                g = h5_file[soup_id_to_key[i]]
                meta = pickle.loads(g.attrs['meta'])
                meta = translateFeature(meta, SpfnDatasetReader.FEATURES_BY_TYPE, SpfnDatasetReader.FEATURES_MAPPING)
                features_data[i] = meta

        if unNormalize:
            gt_points, gt_normals, features_data = unNormalize(gt_points, transforms, normals=gt_normals, features=features_data)
            noisy_points, _, _ = unNormalize(noisy_points, transforms, normals=None, features=[])

        result = {
            'noisy_points': noisy_points,
            'points': gt_points,
            'normals': gt_normals,
            'labels': labels,
            'features': features_data,
            'filename': filename,
            'transforms': transforms
        }

        self.steps_by_set[self.current_set_name] += 1
        
        return result
    
    def finish(self):
        super().finish()
    
    def __iter__(self):
        return super().__iter__()