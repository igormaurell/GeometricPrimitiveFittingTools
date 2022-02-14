from cProfile import label
from lib.makers import *
from lib.utils import filterFeaturesData, face2Primitive

import random
from copy import deepcopy

class DatasetFactory:
    MAKERS_DICT = {
        'default': DefaultDatasetMaker,
        'spfn': SpfnDatasetMaker,
    }

    def __init__(self, parameters):
        formats = parameters.keys()
        assert all([format in DatasetFactory.MAKERS_DICT.keys() for format in formats])
        self.filter_features_groups = []
        self.filter_features_groups_names = []
        self.makers = {}
        for format in formats:
            #improving performance using filter features groups
            if 'filter_features' in parameters[format].keys():
                if parameters[format]['filter_features'] not in self.filter_features_groups:
                    self.filter_features_groups.append(parameters[format]['filter_features'])
                    self.filter_features_groups_names.append([format])
                else:
                    index = self.filter_features_groups.index(parameters[format]['filter_features'])
                    self.filter_features_groups_names[index].append(format)
                parameters[format].pop('filter_features')
            self.makers[format] = DatasetFactory.MAKERS_DICT[format](parameters[format])
        random.seed(1234)
        self.step_num = 0

    def step(self, points, normals=None, labels=None, features_data=[], filename = None):
        for i in range(len(self.filter_features_groups)):
            features_data_curr = features_data
            labels_curr = labels
            if len(features_data) > 0:
                features_data_curr = filterFeaturesData(deepcopy(features_data_curr), self.filter_features_groups[i]['curve_types'], self.filter_features_groups[i]['surface_types'])
                assert labels is not None and labels.shape[0] == points.shape[0]
                labels_curr, features_data_curr = face2Primitive(labels_curr.copy(), features_data_curr['surfaces'])

            for format in self.filter_features_groups_names[i]:
                self.makers[format].step(points, normals=normals, labels=labels_curr, features_data=features_data_curr, filename=filename)
        self.step_num += 1

    def finish(self):
        permutation = random.shuffle(list(range(self.step_num)))
        for maker in self.makers.values():
            maker.finish(permutation)