from abc import abstractmethod
from copy import deepcopy
import random
from math import ceil, floor

class BaseDatasetMaker:

    def __init__(self, parameters):
        self.setParameters(parameters)
        self.reset()
    
    def reset(self):
        self.current_set_name = 'train'
        self.filenames_by_set = {'train': [], 'test': []}

    def setParameters(self, parameters):
        self.dataset_folder_name = parameters['dataset_folder_name'] if 'dataset_folder_name' in parameters.keys() else None
        self.data_folder_name = parameters['data_folder_name'] if 'data_folder_name' in parameters.keys() else None
        self.transform_folder_name = parameters['transform_folder_name'] if 'transform_folder_name' in parameters.keys() else None
        self.normalization_parameters = parameters['normalization'] if 'normalization' in parameters.keys() else None
        self.train_percentage = parameters['train_percentage'] if 'train_percentage' in parameters.keys() else None
        self.filter_features_parameters = parameters['filter_features'] if 'filter_features' in parameters.keys() else None

    def divisionTrainVal(self, permutation=None):
        if len(self.filenames_by_set['train']) > 0 and len(self.filenames_by_set['test']) > 0:
            return self.filenames_by_set['train'], self.filenames_by_set['test']
        elif len(self.filenames_by_set['train']) > 0:
            filenames = self.filenames_by_set['train']
        elif len(self.filenames_by_set['test']) > 0:
            filenames = self.filenames_by_set['test']
        else:
            return [], []
        if permutation is None:
            random.shuffle(filenames)
        else:
            filenames = [filenames[index] for index in permutation]
        n_train = len(filenames)*(self.train_percentage/100.)
        n_train = ceil(n_train) if n_train < 1 else floor(n_train)
        return filenames[:n_train], filenames[n_train:]
    
    def setCurrentSetName(self, set_name):
        assert set_name in self.filenames_by_set.keys()
        self.current_set_name = set_name

    def finish(self, permutation=None):
        self.reset()

    @abstractmethod
    def step(self, points, normals=None, labels=None, features_data=[], filename=None, is_face_labels=False):
        pass