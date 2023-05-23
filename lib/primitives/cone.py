from .base_primitive_surface import BasePrimitiveSurface, rotate
import numpy as np
from math import tan

class Cone(BasePrimitiveSurface):   
    def getPrimitiveType(self):
        return 'cone'
    
    def getColor(self):
        return (0, 255, 0)

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
        self.angle = BasePrimitiveSurface.readParameterOnDict('angle', parameters, old_value=(self.angle if update else None))
        self.apex = BasePrimitiveSurface.readParameterOnDict('apex', parameters, old_value=(self.apex if update else None))

    def toDict(self):
        parameters = super().toDict()
        parameters['type'] = self.getPrimitiveType()
        BasePrimitiveSurface.writeParameterOnDict('location', self.location, parameters)
        BasePrimitiveSurface.writeParameterOnDict('x_axis', self.x_axis, parameters)
        BasePrimitiveSurface.writeParameterOnDict('y_axis', self.y_axis, parameters)
        BasePrimitiveSurface.writeParameterOnDict('z_axis', self.z_axis, parameters)
        BasePrimitiveSurface.writeParameterOnDict('coefficients', self.coefficients, parameters)
        BasePrimitiveSurface.writeParameterOnDict('radius', self.radius, parameters)
        BasePrimitiveSurface.writeParameterOnDict('angle', self.angle, parameters)
        BasePrimitiveSurface.writeParameterOnDict('apex', self.apex, parameters)
        return parameters

    def _computeCorrectPointAndNormal(self, P):
        B = self.apex
        n = self.z_axis
        h = (P - B) @ n
        if h < 0:
            P_new = B.copy()
            n_new = -n
        else:
            radius = h*tan(self.angle)
            P_proj = B + h*n
            P_projP = P - P_proj
            n_orth = P_projP/np.linalg.norm(P_projP, ord=2)
            P_new = P_proj + radius*n_orth
            v_rot = np.cross(n, n_orth)
            n_new = rotate(n_orth, self.angle, v_rot)
        return np.concatenate((P_new, n_new))