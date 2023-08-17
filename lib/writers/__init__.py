# from .spfn_dataset_writer import SpfnDatasetWriter
# from .primitivenet_dataset_writer import PrimitivenetDatasetWriter
# from .parsenet_dataset_writer import ParsenetDatasetWriter
from .ls3dc_dataset_writer import LS3DCDatasetWriter
from .hpnet_dataset_writer import HPNetDatasetWriter

import random

from copy import copy, deepcopy
import ujson

class DatasetWriterFactory:
    WRITERS_DICT = {
        # 'spfn': SpfnDatasetWriter,
        # 'primitivenet': PrimitivenetDatasetWriter,
        # 'parsenet': ParsenetDatasetWriter,
        'ls3dc': LS3DCDatasetWriter,
        'hpnet': HPNetDatasetWriter
    }

    def __init__(self, parameters):
        formats = parameters.keys()
        assert all([format in DatasetWriterFactory.WRITERS_DICT.keys() for format in formats])
        self.writers = {}
        for format in formats:
            self.writers[format] = DatasetWriterFactory.WRITERS_DICT[format](parameters[format])
        random.seed(1234)
        self.step_num = 0

    def getWriterInputTypes(self, writer):
        assert writer in DatasetWriterFactory.WRITERS_DICT
        return DatasetWriterFactory.WRITER_INPUT_TYPES[writer]
    
    def getWriterByFormat(self, format):
        return self.writers[format]

    def setCurrentSetNameAllFormats(self, set_name):
        for writer in self.writers.values():
            writer.setCurrentSetName(set_name)

    def stepAllFormats(self, points, normals=None, labels=None, features_data=[], noisy_points=None, filename=None,
                       features_point_indices=None, mesh=None, **kwargs):
        fn_1 = lambda x: None if x is None else x.copy()
        json_dc = lambda x: None if x is None else ujson.loads(ujson.dumps(x))
        for writer in self.writers.values():
            writer.step(points.copy(), normals=fn_1(normals), labels=fn_1(labels), features_data=json_dc(features_data), 
                        noisy_points=fn_1(noisy_points), filename=copy(filename), features_point_indices=deepcopy(features_point_indices),
                        mesh=deepcopy(mesh), **kwargs)
        self.step_num += 1

    def finishAllFormats(self):
        permutation = random.shuffle(list(range(self.step_num)))
        for writer in self.writers.values():
            writer.finish(permutation)
        self.step_num = 0