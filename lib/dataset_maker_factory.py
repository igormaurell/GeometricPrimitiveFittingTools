from lib.makers import *
import random

from copy import deepcopy

class DatasetMakerFactory:
    MAKERS_DICT = {
        'spfn': SpfnDatasetMaker,
    }

    def __init__(self, parameters):
        formats = parameters.keys()
        assert all([format in DatasetMakerFactory.MAKERS_DICT.keys() for format in formats])
        self.makers = {}
        for format in formats:
            self.makers[format] = DatasetMakerFactory.MAKERS_DICT[format](parameters[format])
        random.seed(1234)
        self.step_num = 0
    
    def getMakerByFormat(self, format):
        return self.makers[format]

    def stepAllFormats(self, points, normals=None, labels=None, features_data=[], filename=None, is_face_labels=False):
        for maker in self.makers.values():
            maker.step(points.copy(), normals=normals.copy(), labels=labels.copy(), features_data=deepcopy(features_data), filename=filename, is_face_labels=is_face_labels)
        self.step_num += 1

    def finishAllFormats(self):
        permutation = random.shuffle(list(range(self.step_num)))
        for maker in self.makers.values():
            maker.finish(permutation)
        self.step_num = 0