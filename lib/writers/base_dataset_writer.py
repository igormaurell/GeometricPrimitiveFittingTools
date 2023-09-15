from abc import abstractmethod
from lib.normalization import normalize
from asGeometryOCCWrapper.geometry.base_geometry import angleDeviation
import random
from math import ceil, floor

class BaseDatasetWriter:

    def __init__(self, parameters):
        self.setParameters(parameters)
        self.reset()
    
    def normalize(self, points, noisy_points, normals, noisy_normals, features_data):
        points_noise_limit = 0.
        normals_noise_limit = 0.
        if 'points_noise' in self.normalization_parameters.keys():
            points_noise_limit = self.normalization_parameters['points_noise']
            self.normalization_parameters['points_noise'] = 0.
        if 'normals_noise' in self.normalization_parameters.keys():
            normals_noise_limit = self.normalization_parameters['normals_noise']
            self.normalization_parameters['normals_noise'] = 0.

        noisy_points = points.copy() if noisy_points is None else noisy_points
        noisy_normals = normals.copy() if noisy_normals is None else noisy_normals
            
        points, normals, features_data, transforms = normalize(points.copy(), self.normalization_parameters, 
                                                                normals=normals.copy(), features=features_data)

        self.normalization_parameters['points_noise'] = points_noise_limit
        noisy_points, _, _, _ = normalize(noisy_points, self.normalization_parameters, normals=normals.copy())
            
        self.normalization_parameters['normals_noise'] = normals_noise_limit
        _, noisy_normals, _, _ = normalize(points.copy(), self.normalization_parameters, normals=noisy_normals)

        return points, noisy_points, normals, noisy_normals, features_data, transforms
    
    def reset(self):
        self.current_set_name = 'train'
        self.filenames_by_set = {'train': [], 'val': []}

    def setParameters(self, parameters):
        self.dataset_folder_name = parameters['dataset_folder_name'] if 'dataset_folder_name' in parameters.keys() else ''
        self.data_folder_name = parameters['data_folder_name'] if 'data_folder_name' in parameters.keys() else ''
        self.transform_folder_name = parameters['transform_folder_name'] if 'transform_folder_name' in parameters.keys() else ''
        self.normalization_parameters = parameters['normalization'] if 'normalization' in parameters.keys() else {}
        self.train_percentage = parameters['train_percentage'] if 'train_percentage' in parameters.keys() else 1.
        self.min_number_points = parameters['min_number_points'] if 'min_number_points' in parameters.keys() else 0
        self.filter_features_parameters = parameters['filter_features'] if 'filter_features' in parameters.keys() else {}
        self.surface_types = self.filter_features_parameters['surface_types'] if 'surface_types' in \
                             self.filter_features_parameters.keys() else None

    def divisionTrainVal(self, permutation=None):
        if len(self.filenames_by_set['val']) > 0:
            return self.filenames_by_set['train'], self.filenames_by_set['val']
        elif len(self.filenames_by_set['train']) > 0:
            if self.train_percentage is None:
                return self.filenames_by_set['train'], []
            filenames = self.filenames_by_set['train']
        elif len(self.filenames_by_set['val']) > 0:
            if self.train_percentage is None:
                return [], self.filenames_by_set['val']
            filenames = self.filenames_by_set['val']
        else:
            return [], []
        if permutation is None:
            random.shuffle(filenames)
        else:
            filenames = [filenames[index] for index in permutation]
        n_train = len(filenames)*(self.train_percentage)
        n_train = ceil(n_train) if n_train < 1 else floor(n_train)
        return filenames[:n_train], filenames[n_train:]
    
    def setCurrentSetName(self, set_name):
        assert set_name in self.filenames_by_set.keys()
        self.current_set_name = set_name

    def finish(self, permutation=None):
        self.reset()

    @abstractmethod
    def step(self, points, normals=None, labels=None, features_data=[], noisy_points=None,
             noisy_normals=None, filename=None, features_point_indices=None, **kwargs):
        pass