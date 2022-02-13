from lib.makers import *

class DatasetFactory:
    MAKERS_DICT = {
        'default': DefaultDatasetMaker,
        'spfn': SpfnDatasetMaker,
    }

    def __init__(self, formats):
        assert formats in DatasetFactory.MAKERS_DICT.keys()
        self.makers = [DatasetFactory.MAKERS_DICT[format] for format in formats]