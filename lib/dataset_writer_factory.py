from lib.writers import *
import random

from copy import deepcopy

class DatasetWriterFactory:
    WRITERS_DICT = {
        'spfn': SpfnDatasetWriter,
    }

    def __init__(self, parameters):
        formats = parameters.keys()
        assert all([format in DatasetWriterFactory.WRITERS_DICT.keys() for format in formats])
        self.writers = {}
        for format in formats:
            self.writers[format] = DatasetWriterFactory.WRITERS_DICT[format](parameters[format])
        random.seed(1234)
        self.step_num = 0
    
    def getWriterByFormat(self, format):
        return self.writers[format]

    def setCurrentSetNameAllFormats(self, set_name):
        for writer in self.writers.values():
            writer.setCurrentSetName(set_name)

    def stepAllFormats(self, points, normals=None, labels=None, features_data=[], filename=None, features_point_indices=None):
        for writer in self.writers.values():
            writer.step(points.copy(), normals=normals.copy(), labels=labels.copy(), features_data=deepcopy(features_data), filename=filename, features_point_indices=features_point_indices)
        self.step_num += 1

    def finishAllFormats(self):
        permutation = random.shuffle(list(range(self.step_num)))
        for writer in self.writers.values():
            writer.finish(permutation)
        self.step_num = 0