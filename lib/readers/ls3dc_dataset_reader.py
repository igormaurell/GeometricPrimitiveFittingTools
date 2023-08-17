import pickle
import h5py
import csv
from os.path import join
import re

from .base_dataset_reader import BaseDatasetReader

from lib.normalization import unNormalize

def decode_string(binary_string):
    return binary_string.decode('utf-8')

def hdf5_group_to_dict(group):
    data_dict = {}
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            data_dict[key] = hdf5_group_to_dict(item)
        elif isinstance(item, h5py.Dataset):
            if item.dtype.kind == 'O':  # Check if the dataset contains string data
                data_dict[key] = decode_string(item[()])
            else:
                data_dict[key] = item[()].tolist()  # Get the dataset's value as a NumPy array
    return data_dict

class LS3DCDatasetReader(BaseDatasetReader):
    def __init__(self, parameters):
        super().__init__(parameters)

        with open(join(self.data_folder_name, 'train_models.csv'), 'r', newline='') as f:
            data = list(csv.reader(f, delimiter=',', quotechar='|'))
            self.filenames_by_set['train'] = data[0] if len(data) > 0 else data
        with open(join(self.data_folder_name, 'test_models.csv'), 'r', newline='') as f:
            data = list(csv.reader(f, delimiter=',', quotechar='|'))
            self.filenames_by_set['val'] = data[0] if len(data) > 0 else data

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
            gt_points = h5_file['gt_points'][()] if 'gt_points' in h5_file.keys() else None
            noisy_points = h5_file['noisy_points'][()] if 'noisy_points' in h5_file.keys() else None
            if noisy_points is None and gt_points is not None:
                noisy_points = gt_points.copy()
            gt_normals = h5_file['gt_normals'][()] if 'gt_normals' in h5_file.keys() else None
            labels = h5_file['gt_labels'][()] if 'gt_labels' in h5_file.keys() else None
            gt_indices = h5_file['gt_indices'][()] if 'gt_indices' in h5_file.keys() else None
            matching = h5_file['matching'][()] if 'matching' in h5_file.keys() else None

            found_soup_ids = []
            soup_id_to_key = {}
            soup_prog = re.compile('feature_([0-9]+)$')
            for key in list(h5_file.keys()):
                m = soup_prog.match(key)
                if m is not None:
                    soup_id = int(m.group(1))
                    found_soup_ids.append(soup_id)
                    soup_id_to_key[soup_id] = key

            max_size = max(found_soup_ids) + 1 if len(found_soup_ids) > 0 else 0
            features_data = [None]*max_size  
            found_soup_ids.sort()
            for i in found_soup_ids:
                g = h5_file[soup_id_to_key[i]]
                features_data[i] = hdf5_group_to_dict(g['parameters'])
            
            if unormalize:
                gt_points, gt_normals, features_data = unNormalize(gt_points, transforms, normals=gt_normals, features=features_data)
                noisy_points, _, _ = unNormalize(noisy_points, transforms, normals=None, features=[])

        result = {
            'noisy_points': noisy_points,
            'points': gt_points,
            'normals': gt_normals,
            'labels': labels,
            'features_data': features_data,
            'filename': filename,
            'transforms': transforms
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