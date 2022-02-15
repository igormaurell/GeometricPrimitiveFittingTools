from abc import abstractmethod
from copy import deepcopy
import random
from math import ceil, floor

class BaseDatasetMaker:

    def __init__(self, parameters):
        self.dataset_folder_name = parameters['dataset_folder_name'] if 'dataset_folder_name' in parameters.keys() else None
        self.data_folder_name = parameters['data_folder_name'] if 'data_folder_name' in parameters.keys() else None
        self.transform_folder_name = parameters['transform_folder_name'] if 'transform_folder_name' in parameters.keys() else None
        self.normalization_parameters = parameters['normalization'] if 'normalization' in parameters.keys() else None
        self.train_percentage = parameters['train_percentage'] if 'train_percentage' in parameters.keys() else None
        self.filenames = []

    def divideTrainVal(self, permutation=None):
        if permutation is None:
            filenames = deepcopy(self.filenames)
            random.shuffle(filenames)
        else:
            filenames = [self.filenames[index] for index in permutation]
        n_train = len(filenames)*(self.train_percentage/100.)
        n_train = ceil(n_train) if n_train < 1 else floor(n_train)
        return filenames[:n_train], filenames[n_train:]

    @abstractmethod
    def step(self, points, normals=None, labels=None, features_data=[], filename=None):
        pass

    @abstractmethod
    def finish(self, permutation=None):
        pass