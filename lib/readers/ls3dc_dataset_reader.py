import pickle
import h5py
import csv
from os.path import join, exists
import re
import numpy as np

from .base_dataset_reader import BaseDatasetReader

from lib.normalization import applyTransforms, cubeRescale
from lib.fitting_func import FittingFunctions

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
        
        if exists(join(self.data_folder_name, 'train_models.csv')):
            with open(join(self.data_folder_name, 'train_models.csv'), 'r', newline='') as f:
                data = list(csv.reader(f, delimiter=',', quotechar='|'))
                data = data[0] if len(data) > 0 else data
                data = [d[:d.rfind('.')] for d in data]
                self.filenames_by_set['train'] = data
        else:
            self.filenames_by_set['train'] = []

        if exists(join(self.data_folder_name, 'test_models.csv')):  
            with open(join(self.data_folder_name, 'test_models.csv'), 'r', newline='') as f:
                data = list(csv.reader(f, delimiter=',', quotechar='|'))
                data = data[0] if len(data) > 0 else data
                data = [d[:d.rfind('.')] for d in data]
                self.filenames_by_set['val'] = data
        else:
            self.filenames_by_set['val'] = []

    def step(self, **kwargs):
        assert self.current_set_name in self.filenames_by_set.keys()

        index = self.steps_by_set[self.current_set_name]%len(self.filenames_by_set[self.current_set_name])
        filename = self.filenames_by_set[self.current_set_name][index]

        data_file_path = join(self.data_folder_name, f'{filename}.h5')
        transforms_file_path = join(self.transform_folder_name, f'{filename}.pkl')

        with open(transforms_file_path, 'rb') as pkl_file:
            transforms = pickle.load(pkl_file)

        with h5py.File(data_file_path, 'r') as h5_file:
            gt_points = h5_file['gt_points'][()].astype(np.float32) if 'gt_points' in h5_file.keys() else None
            noisy_points = h5_file['noisy_points'][()].astype(np.float32) if 'noisy_points' in h5_file.keys() else None
            if noisy_points is None and gt_points is not None:
                noisy_points = gt_points.copy()
            gt_normals = h5_file['gt_normals'][()].astype(np.float32) if 'gt_normals' in h5_file.keys() else None
            noisy_normals = h5_file['noisy_normals'][()].astype(np.float32) if 'noisy_normals' in h5_file.keys() else None
            if noisy_normals is None and gt_normals is not None:
                noisy_normals = gt_normals.copy()
            labels = h5_file['gt_labels'][()].astype(np.int32) if 'gt_labels' in h5_file.keys() else None
            gt_indices = h5_file['gt_indices'][()].astype(np.int32) if 'gt_indices' in h5_file.keys() else None
            matching = h5_file['matching'][()].astype(np.int32) if 'matching' in h5_file.keys() else None
            global_indices = h5_file['global_indices'][()].astype(np.int32) if 'global_indices' in h5_file.keys() else None

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
            points_scale = None
            for i in found_soup_ids:
                g = h5_file[soup_id_to_key[i]]
                feature = hdf5_group_to_dict(g['parameters'])
                if not self.use_data_primitives:
                    tp = feature['type']
                    mask = labels==i
                    points = noisy_points if not self.fit_noisy_points else gt_points
                    normals = noisy_normals if not self.fit_noisy_normals else gt_normals
                    if points_scale is None:
                        _, _, points_scale = cubeRescale(points.copy())
                    feature = FittingFunctions.fit(tp, points[mask], normals[mask], scale=1/points_scale)
                features_data[i] = feature
            
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
            'transforms': transforms
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