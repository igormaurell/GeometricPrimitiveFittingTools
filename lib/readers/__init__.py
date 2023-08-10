# from .spfn_dataset_reader import SpfnDatasetReader
# from .primitivenet_dataset_reader import PrimitivenetDatasetReader
from .ls3dc_dataset_reader import LS3DCDatasetReader
from .hpnet_dataset_reader import HPNetDatasetReader

class DatasetReaderFactory:
    READERS_DICT = {
        #'spfn': SpfnDatasetReader,
        'ls3dc': LS3DCDatasetReader,
        #'primitivenet': PrimitivenetDatasetReader,
        'hpnet': HPNetDatasetReader
    }

    def __init__(self, parameters):
        formats = parameters.keys()
        assert all([format in DatasetReaderFactory.READERS_DICT.keys() for format in formats])
        self.readers = {}
        for format in formats:
            self.readers[format] = DatasetReaderFactory.READERS_DICT[format](parameters[format])
    
    def getReaderByFormat(self, format):
        return self.readers[format]