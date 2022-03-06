from lib import primitives
from lib.primitives import *

class PrimitiveSurfaceFactory:
    PRIMITIVES_DICT = {
        'plane': Plane,
        'cone': Cone,
        'sphere': Sphere,
        'cylinder': Cylinder,
    }

    @staticmethod
    def getTypeLabel(type):
        type = type.lower()
        assert type in PrimitiveSurfaceFactory.PRIMITIVES_DICT
        return list(PrimitiveSurfaceFactory.PRIMITIVES_DICT.keys()).index(type)

    @staticmethod
    def getPrimitiveClass(type):
        type = type.lower()
        assert type in PrimitiveSurfaceFactory.PRIMITIVES_DICT
        return PrimitiveSurfaceFactory.PRIMITIVES_DICT[type]
    
    @staticmethod
    def primitiveFromDict(parameters: dict):
        assert 'type' in parameters
        primitive = PrimitiveSurfaceFactory.getPrimitiveClass(parameters['type'])()
        primitive.fromDict(parameters)
        return primitive
    
    @staticmethod
    def primitivesFromListOfDicts(parameters_list: list):
        primitives = []
        for parameters in parameters_list:
            primitives.append(PrimitiveSurfaceFactory.primitiveFromDict(parameters))
        return primitives
    
    @staticmethod
    def dictFromPrimitive(primitive):
        assert issubclass(primitive, PrimitiveSurface)
        parameters = primitive.fromDict()
        return parameters

    @staticmethod
    def listOfDictsFromPrimitives(primitives):
        parameters_list = []
        for primitive in primitives:
            parameters_list.append(PrimitiveSurfaceFactory.dictFromPrimitive(primitive))
        return parameters_list