# from .spfn_dataset_reader import SpfnDatasetReader
from .primitivenet_dataset_reader import PrimitivenetDatasetReader
from .parsenet_dataset_reader import ParsenetDatasetReader
from .ls3dc_dataset_reader import LS3DCDatasetReader
from .hpnet_dataset_reader import HPNetDatasetReader

class PredAndGTDatasetReader:
    def __init__(self, pred_reader, gt_reader):
        self.pred_reader = pred_reader
        self.gt_reader = gt_reader

    def __iter__(self):
        self.pred_iter = self.pred_reader.__iter__()
        self.gt_iter = self.gt_reader.__iter__()
        return self

    def __next__(self):
        try:
            pred = self.pred_iter.__next__()
            gt = self.gt_iter.__next__()
            return pred, gt
        except StopIteration:
            raise StopIteration
    
    def __len__(self):
        return len(self.pred_reader.filenames_by_set[self.current_set_name])

class DatasetReaderFactory:
    READERS_DICT = {
        #'spfn': SpfnDatasetReader,
        'parsenet': ParsenetDatasetReader,
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