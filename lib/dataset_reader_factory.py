from lib.readers.spfn_dataset_reader import SpfnDatasetReader
from lib.readers.primitivenet_dataset_reader import PrimitivenetDatasetReader
from lib.readers.ls3dc_dataset_reader import LS3DCDatasetReader

class DatasetReaderFactory:
    READERS_DICT = {
        'spfn': SpfnDatasetReader,
        'ls3dc': LS3DCDatasetReader,
        'primitivenet': PrimitivenetDatasetReader,
    }

    def __init__(self, parameters):
        formats = parameters.keys()
        assert all([format in DatasetReaderFactory.READERS_DICT.keys() for format in formats])
        self.readers = {}
        for format in formats:
            self.readers[format] = DatasetReaderFactory.READERS_DICT[format](parameters[format])
    
    def getReaderByFormat(self, format):
        return self.readers[format]