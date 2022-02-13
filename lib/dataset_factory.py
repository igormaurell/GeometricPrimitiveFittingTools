from lib.makers import *

class DatasetFactory:
    MAKERS_DICT = {
        'default': DefaultDatasetMaker,
        'spfn': SpfnDatasetMaker,
    }

    def __init__(self, parameters):
        formats = parameters.keys()
        assert formats in DatasetFactory.MAKERS_DICT.keys()
        self.filter_features_groups = []
        self.filter_features_groups_names = []
        self.makers = {}
        for format in formats:
            #improving performance using filter features groups
            if 'filter_features' in parameters.keys():
                if parameters[format]['filter_features'] not in self.filter_featuress_groups:
                    self.filter_features_groups.append([parameters[format]['filter_features']])
                    self.filter_features_groups_names.append([format])
                else:
                    index = self.filter_features_groups.index(parameters[format]['filter_features'])
                    self.filter_features_groups[index].append(parameters[format]['filter_features'])
                    self.filter_features_groups_names[index].append(format)
                parameters[format].pop('filter_features')
            self.makers[format] = DatasetFactory.MAKERS_DICT[format](parameters[format])