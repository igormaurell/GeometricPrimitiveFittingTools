from lib.readers import *

class DatasetReaderFactory:
    READERS_DICT = {
        'spfn': SpfnDatasetReader,
        'ls3dc': LS3DCDatasetReader,
        'primitivenet': PrimitivenetDatasetReader,
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