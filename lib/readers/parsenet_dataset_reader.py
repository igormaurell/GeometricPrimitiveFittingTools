import pickle
import h5py
from os.path import join, exists
import numpy as np
import torch
import statistics as stats

from .base_dataset_reader import BaseDatasetReader

from lib.normalization import applyTransforms, cubeRescale
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
    data['points'] = h5_file['points'][()].astype(np.float32) if 'points' in h5_file.keys() else None
    data['normals'] = h5_file['normals'][()].astype(np.float32) if 'normals' in h5_file.keys() else None
    data['labels'] = h5_file['labels'][()].astype(np.int32) if 'labels' in h5_file.keys() else None
    data['prim'] = h5_file['prim'][()].astype(np.int32) if 'prim' in h5_file.keys() else None
    data['gt_indices'] = h5_file['gt_indices'][()].astype(np.int32) if 'gt_indices' in h5_file.keys() else None
    data['matching'] = h5_file['matching'][()].astype(np.int32) if 'matching' in h5_file.keys() else None
    data['global_indices'] = h5_file['global_indices'][()].astype(np.int32) if 'global_indices' in h5_file.keys() else None
    return data

def to_one_hot(target, maxx=50, device_id=0):
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target.astype(np.int64))
    N = target.shape[0]
    target_one_hot = torch.zeros((N, maxx))

    target_one_hot = target_one_hot
    target_t = target.unsqueeze(1)
    target_one_hot = target_one_hot.scatter_(1, target_t.long(), 1)
    return target_one_hot

def guard_exp(x, max_value=75, min_value=-75):
    x = torch.clamp(x, max=max_value, min=min_value)
    return torch.exp(x)

def labels_to_one_hot(labels, num_classes):
    one_hot = np.zeros((len(labels), num_classes), dtype=np.int64)
    valid_labels_mask = labels >= 0
    one_hot[valid_labels_mask, labels[valid_labels_mask]] = 1
    return one_hot

def weights_normalize(weights, bw):
    """
    Assuming that weights contains dot product of embedding of a
    points with embedding of cluster center, we want to normalize
    these weights to get probabilities. Since the clustering is
    gotten by mean shift clustering, we use the same kernel to compute
    the probabilities also.
    """
    prob = guard_exp(weights / (bw ** 2) / 2)
    prob = prob / torch.sum(prob, 0, keepdim=True)

    # This is to avoid numerical issues
    if weights.shape[0] == 1:
        return prob

    # This is done to ensure that max probability is 1 at the center.
    # this will be helpful for the spline fitting network
    prob = prob - torch.min(prob, 1, keepdim=True)[0]
    prob = prob / (torch.max(prob, 1, keepdim=True)[0] + np.finfo(np.float32).eps)
    return prob


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

            assert exists(path), f'{path} file does not exist.'

            with open(path, 'r') as txt_file:
                filenames = txt_file.read().split('\n')
                if len(filenames[-1].strip()) == 0:
                    filenames.pop(-1)
        else:
            filenames = []

        for i, filename in enumerate(filenames):
            self.data_by_set[set_name][filename] = get_data_at_index(data, i) 
            
        self.filenames_by_set[set_name] = filenames

    def __init__(self, parameters):
        self.data_by_set = {}
        super().__init__(parameters)

        self.read_data('train')
        self.read_data('val')

    def step(self, **kwargs):
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
        gt_indices = data['gt_indices'] if 'gt_indices' in data.keys() else None
        matching = data['matching'] if 'matching' in data.keys() else None
        global_indices = data['global_indices'] if 'global_indices' in data.keys() else None

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
            types = prim[indices]
            tp_id = stats.mode(types)

            tp = ParsenetDatasetReader.PRIMITIVES_MAP[tp_id]
            if points_scale is None:
                _, _, points_scale = cubeRescale(points.copy())
            feature = FittingFunctions.fit(tp, points[indices], normals[indices], scale=1./points_scale)
                        
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