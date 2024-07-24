from abc import abstractmethod

class DatasetReaderIterator:
    def __init__(self, dataset_reader):
        self.dr = dataset_reader

    def __next__(self):
        if self.dr.steps_by_set[self.dr.current_set_name] < len(self.dr.filenames_by_set[self.dr.current_set_name]):
            return self.dr.step()
        raise StopIteration
    
    def __len__(self):
        return len(self.dr)

class BaseDatasetReader:

    def __init__(self, parameters):
        self.setParameters(parameters)
        
        self.reset()
        
    def setParameters(self, parameters):
        self.dataset_folder_name = parameters['dataset_folder_name'] if 'dataset_folder_name' in parameters.keys() else None
        self.data_folder_name = parameters['data_folder_name'] if 'data_folder_name' in parameters.keys() else None
        self.transform_folder_name = parameters['transform_folder_name'] if 'transform_folder_name' in parameters.keys() else None
        self.use_data_primitives = parameters['use_data_primitives'] if 'use_data_primitives' in parameters.keys() else True
        self.fit_noisy_points = parameters['fit_noisy_points'] if 'fit_noisy_points' in parameters.keys() else False
        self.fit_noisy_normals = parameters['fit_noisy_normals'] if 'fit_noisy_normals' in parameters.keys() else False
        self.unnormalize = parameters['unnormalize'] if 'unnormalize' in parameters.keys() else False

    def reset(self):
        self.current_set_name = 'train'
        self.filenames_by_set = {'train': [], 'val': []}
        self.steps_by_set = {'train': 0, 'val': 0}
    
    def setCurrentSetName(self, set_name):
        assert set_name in self.filenames_by_set.keys()
        self.current_set_name = set_name

    def __len__(self):
        return len(self.filenames_by_set[self.current_set_name])

    def finish(self):
        self.reset()

    def __iter__(self):
        return DatasetReaderIterator(self)

    @abstractmethod
    def step(self, unormalize=True, **kwargs):
        pass   