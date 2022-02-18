from abc import abstractmethod

class BaseDatasetReader:

    def __init__(self, parameters):
        self.setParameters(parameters)
        
        self.reset()

    def setParameters(self, parameters):
        self.dataset_folder_name = parameters['dataset_folder_name'] if 'dataset_folder_name' in parameters.keys() else None
        self.data_folder_name = parameters['data_folder_name'] if 'data_folder_name' in parameters.keys() else None
        self.transform_folder_name = parameters['transform_folder_name'] if 'transform_folder_name' in parameters.keys() else None

    def reset(self):
        self.current_set_name = 'train'
        self.filenames_by_set = {'train': [], 'test': []}
        self.steps_by_set = {'train': 0, 'test': 0}
    
    def setCurrentSetName(self, set_name):
        assert set_name in self.filenames_by_set.keys()
        self.current_set_name = set_name

    def __len__(self):
        return len(self.filenames_by_set[self.current_set_name])

    def finish(self):
        self.reset()

    @abstractmethod
    def step(self):
        pass

   