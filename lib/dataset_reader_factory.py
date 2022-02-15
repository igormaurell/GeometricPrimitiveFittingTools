from lib.makers import *
from lib.readers.spfn_dataset_reader import SpfnDatasetReader

class DatasetReaderFactory:
    READERS_DICT = {
        'spfn': SpfnDatasetReader,
    }

    def __init__(self, parameters):
        formats = parameters.keys()
        assert all([format in DatasetReaderFactory.READERS_DICT.keys() for format in formats])
        self.readers = {}
        self.step_num = {}
        for format in formats:
            self.readers[format] = DatasetReaderFactory.READERS_DICT[format](parameters[format])
            self.step_num[format] = 0

    def step(self, format, set_name='train'):
        assert format in self.readers.keys()
        result = self.readers[format].step(set_name=set_name)
        self.step_num[format] += 1
        return result

    def finish(self, format):
        assert format in self.readers.keys()
        self.readers[format].finish()
        self.step_num[format] = 0