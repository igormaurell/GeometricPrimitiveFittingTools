from .base_primitive_surface import BasePrimitiveSurface
import numpy as np

class Sphere(BasePrimitiveSurface):    
    def getPrimitiveType(self):
        return 'sphere'
    
    def getColor(self):
        return (255, 255, 0)

    def __init__(self, parameters: dict = {}):
        super().__init__(parameters=parameters)

    def fromDict(self, parameters: dict, update=False):
        super().fromDict(parameters, update=update)
        self.location = BasePrimitiveSurface.readParameterOnDict('location', parameters, old_value=(self.location if update else None))
        self.x_axis =  BasePrimitiveSurface.readParameterOnDict('x_axis', parameters, old_value=(self.x_axis if update else None))
        self.y_axis = BasePrimitiveSurface.readParameterOnDict('y_axis', parameters, old_value=(self.y_axis if update else None))
        self.z_axis = BasePrimitiveSurface.readParameterOnDict('z_axis', parameters, old_value=(self.z_axis if update else None))
        self.coefficients = BasePrimitiveSurface.readParameterOnDict('coefficients', parameters, old_value=(self.coefficients if update else None))
        self.radius = BasePrimitiveSurface.readParameterOnDict('radius', parameters, old_value=(self.radius if update else None))
    
    def toDict(self):
        parameters = super().toDict()
        parameters['type'] = self.getPrimitiveType()
        BasePrimitiveSurface.writeParameterOnDict('location', self.location, parameters)
        BasePrimitiveSurface.writeParameterOnDict('x_axis', self.x_axis, parameters)
        BasePrimitiveSurface.writeParameterOnDict('y_axis', self.y_axis, parameters)
        BasePrimitiveSurface.writeParameterOnDict('z_axis', self.z_axis, parameters)
        BasePrimitiveSurface.writeParameterOnDict('coefficients', self.coefficients, parameters)
        BasePrimitiveSurface.writeParameterOnDict('radius', self.radius, parameters)
        return parameters
    
    def _computeCorrectPointAndNormal(self, P):
        A = self.location
        PA = P - A
        n_new = PA/np.linalg.norm(PA, ord=2)
        P_new = A + self.radius*n_new
        return np.concatenate((P_new, n_new))