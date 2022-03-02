from .primitive_surface import PrimitiveSurface
from ..deviation import deviationPointCone

class Cone(PrimitiveSurface):
    def __init__(self, parameters: dict = {}):
        self.fromDict(parameters)

    def fromDict(self, parameters: dict):
        super().fromDict(parameters)
        self.location = PrimitiveSurface.readParameterOnDict('location', parameters)
        self.x_axis =  PrimitiveSurface.readParameterOnDict('x_axis', parameters)
        self.y_axis = PrimitiveSurface.readParameterOnDict('y_axis', parameters)
        self.z_axis = PrimitiveSurface.readParameterOnDict('z_axis', parameters)
        self.coefficients = PrimitiveSurface.readParameterOnDict('coefficients', parameters)
        self.radius = PrimitiveSurface.readParameterOnDict('radius', parameters)
        self.angle = PrimitiveSurface.readParameterOnDict('angle', parameters)
        self.apex = PrimitiveSurface.readParameterOnDict('apex', parameters)
    
    def toDict(self):
        parameters = super().toDict()
        PrimitiveSurface.readParameterOnDict('location', self.location, parameters)
        PrimitiveSurface.readParameterOnDict('x_axis', self.x_axis, parameters)
        PrimitiveSurface.readParameterOnDict('y_axis', self.y_axis, parameters)
        PrimitiveSurface.readParameterOnDict('z_axis', self.z_axis, parameters)
        PrimitiveSurface.readParameterOnDict('coefficients', self.coefficients, parameters)
        PrimitiveSurface.readParameterOnDict('radius', self.radius, parameters)
        PrimitiveSurface.readParameterOnDict('angle', self.angle, parameters)
        PrimitiveSurface.readParameterOnDict('apex', self.apex, parameters)

    