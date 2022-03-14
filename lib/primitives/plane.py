from .base_primitive_surface import BasePrimitiveSurface
from lib.utils import angleVectors
import numpy as np

class Plane(BasePrimitiveSurface):
    def getPrimitiveType(self):
        return 'plane'
    
    def getColor(self):
        return (255, 0, 0)

    def __init__(self, parameters: dict = {}):
        super().__init__(parameters=parameters)

    def fromDict(self, parameters: dict, update=False):
        super().fromDict(parameters, update=update)
        self.location = BasePrimitiveSurface.readParameterOnDict('location', parameters, old_value=(self.location if update else None))
        self.x_axis =  BasePrimitiveSurface.readParameterOnDict('x_axis', parameters, old_value=(self.x_axis if update else None))
        self.y_axis = BasePrimitiveSurface.readParameterOnDict('y_axis', parameters, old_value=(self.y_axis if update else None))
        self.z_axis = BasePrimitiveSurface.readParameterOnDict('z_axis', parameters, old_value=(self.z_axis if update else None))
        self.coefficients = BasePrimitiveSurface.readParameterOnDict('coefficients', parameters, old_value=(self.coefficients if update else None))    
    
    def toDict(self):
        parameters = super().toDict()
        parameters['type'] = self.getPrimitiveType()
        BasePrimitiveSurface.readParameterOnDict('location', self.location, parameters)
        BasePrimitiveSurface.readParameterOnDict('x_axis', self.x_axis, parameters)
        BasePrimitiveSurface.readParameterOnDict('y_axis', self.y_axis, parameters)
        BasePrimitiveSurface.readParameterOnDict('z_axis', self.z_axis, parameters)
        BasePrimitiveSurface.readParameterOnDict('coefficients', self.coefficients, parameters)
    
    def _computeCorrectPointAndNormal(self, P):
        A = self.location
        n_new = self.z_axis
        d = (P - A) @ n_new
        P_new = P - d*n_new
        return np.concatenate((P_new, n_new))
        