from .primitive_surface import PrimitiveSurface
from lib.utils import angleVectors
import numpy as np

class Plane(PrimitiveSurface):
    def getPrimitiveType(self):
        return 'plane'
    
    def getColor(self):
        return (255, 0, 0)

    def __init__(self, parameters: dict = {}):
        super().__init__()
        self.location = None
        self.x_axis =  None
        self.y_axis = None
        self.z_axis = None
        self.coefficients = None

    def fromDict(self, parameters: dict, update=False):
        super().fromDict(parameters, update=update)
        self.location = PrimitiveSurface.readParameterOnDict('location', parameters, old_value=(self.location if update else None))
        self.x_axis =  PrimitiveSurface.readParameterOnDict('x_axis', parameters, old_value=(self.x_axis if update else None))
        self.y_axis = PrimitiveSurface.readParameterOnDict('y_axis', parameters, old_value=(self.y_axis if update else None))
        self.z_axis = PrimitiveSurface.readParameterOnDict('z_axis', parameters, old_value=(self.z_axis if update else None))
        self.coefficients = PrimitiveSurface.readParameterOnDict('coefficients', parameters, old_value=(self.coefficients if update else None))    
    
    def toDict(self):
        parameters = super().toDict()
        parameters['type'] = self.getPrimitiveType()
        PrimitiveSurface.readParameterOnDict('location', self.location, parameters)
        PrimitiveSurface.readParameterOnDict('x_axis', self.x_axis, parameters)
        PrimitiveSurface.readParameterOnDict('y_axis', self.y_axis, parameters)
        PrimitiveSurface.readParameterOnDict('z_axis', self.z_axis, parameters)
        PrimitiveSurface.readParameterOnDict('coefficients', self.coefficients, parameters)
    
    def _computeCorrectPointAndNormal(self, P):
        A = self.location
        n_new = self.z_axis
        d = (P - A) @ n_new
        P_new = P - d
        return np.concatenate((P_new, n_new))

    def computeCorrectPointsAndNormals(self, points):
        points_normals = np.array([self._computeCorrectPointAndNormal(P) for P in points], dtype=points.dtype)
        return points_normals[:, :3], points_normals[:, 3:]

    def computeErrors(self, points, normals):
        assert len(points) > 0 and len(normals) > 0
        new_points, new_normals = self.computeCorrectPointsAndNormals(points)
        distances = np.array([np.linalg.norm(P - points[i], ord=2) for i, P in enumerate(new_points)], dtype=new_points.dtype)
        angles = np.array([angleVectors(n, normals[i]) for i, n in enumerate(new_normals)], dtype=new_normals.dtype)
        result = {'distances': distances, 'angles': angles}
        return result