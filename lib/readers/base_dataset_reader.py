from abc import abstractmethod

class BaseDatasetReader:

    def __init__(self, parameters):
        self.dataset_folder_name = parameters['dataset_folder_name'] if 'dataset_folder_name' in parameters.keys() else None
        self.data_folder_name = parameters['data_folder_name'] if 'data_folder_name' in parameters.keys() else None
        self.transform_folder_name = parameters['transform_folder_name'] if 'transform_folder_name' in parameters.keys() else None
        self.filenames = []

    @abstractmethod
    def step(self, filename):
        pass