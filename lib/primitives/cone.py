import numpy as np

from .primitive_surface import PrimitiveSurface
from lib.deviation import deviationPointCone

class Cone(PrimitiveSurface):   
    def getPrimitiveType(self):
        return 'cone'

    def __init__(self, parameters: dict = {}):
        super().__init__()
        self.location = None
        self.x_axis =  None
        self.y_axis = None
        self.z_axis = None
        self.coefficients = None
        self.radius = None
        self.angle = None
        self.apex = None

    def fromDict(self, parameters: dict, update=False):
        super().fromDict(parameters, update=update)
        self.location = PrimitiveSurface.readParameterOnDict('location', parameters, old_value=(self.vert_indices if update else None))
        self.x_axis =  PrimitiveSurface.readParameterOnDict('x_axis', parameters, old_value=(self.x_axis if update else None))
        self.y_axis = PrimitiveSurface.readParameterOnDict('y_axis', parameters, old_value=(self.y_axis if update else None))
        self.z_axis = PrimitiveSurface.readParameterOnDict('z_axis', parameters, old_value=(self.z_axis if update else None))
        self.coefficients = PrimitiveSurface.readParameterOnDict('coefficients', parameters, old_value=(self.coefficients if update else None))
        self.radius = PrimitiveSurface.readParameterOnDict('radius', parameters, old_value=(self.radius if update else None))
        self.angle = PrimitiveSurface.readParameterOnDict('angle', parameters, old_value=(self.angle if update else None))
        self.apex = PrimitiveSurface.readParameterOnDict('apex', parameters, old_value=(self.apex if update else None))

    def toDict(self):
        parameters = super().toDict()
        parameters['type'] = self.getPrimitiveType()
        PrimitiveSurface.readParameterOnDict('location', self.location, parameters)
        PrimitiveSurface.readParameterOnDict('x_axis', self.x_axis, parameters)
        PrimitiveSurface.readParameterOnDict('y_axis', self.y_axis, parameters)
        PrimitiveSurface.readParameterOnDict('z_axis', self.z_axis, parameters)
        PrimitiveSurface.readParameterOnDict('coefficients', self.coefficients, parameters)
        PrimitiveSurface.readParameterOnDict('radius', self.radius, parameters)
        PrimitiveSurface.readParameterOnDict('angle', self.angle, parameters)
        PrimitiveSurface.readParameterOnDict('apex', self.apex, parameters)

    def computeErrors(self, points, normals):
        deviation_function = deviationPointCone
        args = (self.location, self.z_axis, self.apex, self.radius, self.angle,)
        return PrimitiveSurface.genericComputeErrors(points, normals, deviation_function, args)
