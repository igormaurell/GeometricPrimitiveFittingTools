from abc import abstractmethod

class PrimitiveSurface:
    @staticmethod
    def readParameterOnDict(key, d):
        return d[key] if key in d.keys else None
    
    @staticmethod
    def writeParameterOnDict(key, value, d):
        if value is not None:
            d[key] = value

    @abstractmethod
    def __init__(self, parameters: dict = {}):
        pass

    def fromDict(self, parameters: dict):
        self.vert_indices = PrimitiveSurface.readParameterOnDict('vert_indices', parameters)
        self.vert_parameters = PrimitiveSurface.readParameterOnDict('vert_parameters', parameters)
        self.face_indices = PrimitiveSurface.readParameterOnDict('face_indices', parameters)
    
    def toDict(self):
        parameters = {}
        PrimitiveSurface.writeParameterOnDict('vert_indices', self.vert_indices, parameters)
        PrimitiveSurface.writeParameterOnDict('vert_parameters', self.vert_parameters, parameters)
        PrimitiveSurface.writeParameterOnDict('face_indices', self.face_indices, parameters)
        return parameters