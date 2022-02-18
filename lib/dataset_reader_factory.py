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
        self.current_format = formats[0]
    
    def setCurrentFormat(self, format):
        self.current_format = format
    
    def setCurrentSetName(self, set_name):
        self.readers[self.current_format].setCurrentSetName(set_name)
    
    def setCurrentFormatAndSetName(self, format, set_name):
        self.setCurrentFormat(format)
        self.setCurrentSetName(set_name)
    
    def __len__(self):
        return len(self.readers[self.current_format])
    
    def step(self):
        assert format in self.readers.keys()
        result = self.readers[self.current_format].step()
        self.step_num[self.current_format] += 1
        return result

    def finish(self, format):
        assert format in self.readers.keys()
        self.readers[format].finish()
        self.step_num[format] = 0