from .primitive_surface import PrimitiveSurface
from lib.utils import angleVectors
import numpy as np

class Sphere(PrimitiveSurface):    
    def getPrimitiveType(self):
        return 'sphere'
    
    def getColor(self):
        return (255, 255, 0)

    def __init__(self, parameters: dict = {}):
        super().__init__(parameters=parameters)

    def fromDict(self, parameters: dict, update=False):
        super().fromDict(parameters, update=update)
        self.location = PrimitiveSurface.readParameterOnDict('location', parameters, old_value=(self.location if update else None))
        self.x_axis =  PrimitiveSurface.readParameterOnDict('x_axis', parameters, old_value=(self.x_axis if update else None))
        self.y_axis = PrimitiveSurface.readParameterOnDict('y_axis', parameters, old_value=(self.y_axis if update else None))
        self.z_axis = PrimitiveSurface.readParameterOnDict('z_axis', parameters, old_value=(self.z_axis if update else None))
        self.coefficients = PrimitiveSurface.readParameterOnDict('coefficients', parameters, old_value=(self.coefficients if update else None))
        self.radius = PrimitiveSurface.readParameterOnDict('radius', parameters, old_value=(self.radius if update else None))
    
    def toDict(self):
        parameters = super().toDict()
        parameters['type'] = self.getPrimitiveType()
        PrimitiveSurface.readParameterOnDict('location', self.location, parameters)
        PrimitiveSurface.readParameterOnDict('x_axis', self.x_axis, parameters)
        PrimitiveSurface.readParameterOnDict('y_axis', self.y_axis, parameters)
        PrimitiveSurface.readParameterOnDict('z_axis', self.z_axis, parameters)
        PrimitiveSurface.readParameterOnDict('coefficients', self.coefficients, parameters)
        PrimitiveSurface.readParameterOnDict('radius', self.radius, parameters)
    
    def _computeCorrectPointAndNormal(self, P):
        A = self.location
        PA = P - A
        n_new = PA/np.linalg.norm(PA, ord=2)
        P_new = A + self.radius*n_new
        return np.concatenate((P_new, n_new))