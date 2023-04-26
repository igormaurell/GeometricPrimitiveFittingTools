from abc import abstractmethod
import numpy as np

'''VECTOR'''
def rotate(array, theta, axis):
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    
    R = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
    
    return R @ array

def angleVectors(n1, n2):
    n1_unit = n1/np.linalg.norm(n1)
    n2_unit = n2/np.linalg.norm(n2)
    return np.arccos(np.clip(np.dot(n1_unit, n2_unit), -1.0, 1.0))
class BasePrimitiveSurface:
    @staticmethod
    def readParameterOnDict(key, d, old_value=None):
        r = None if key not in d.keys() and old_value is None else d[key]
        if type(r) is list:
            r = np.array(r)
        return r
    
    @staticmethod
    def writeParameterOnDict(key, value, d):
        if value is not None:
            d[key] = value
    
    @staticmethod
    def genericComputeErrors(points, normals, deviation_function, primitive_args):
        assert len(points) == len(normals)
        distances = np.zeros(len(points))
        angles = np.zeros(len(points))
        for i in range(len(points)):
            distance, angle = deviation_function(points[i], normals[i], *primitive_args)
            distances[i] = distance
            angles[i] = angle
        result = {'distances': distances, 'angles': angles}
        return result

    @abstractmethod
    def getPrimitiveType(self):
        pass

    @abstractmethod
    def getColor(self):
        pass

    @abstractmethod
    def _computeCorrectPointAndNormal(self, P):
        pass

    def __init__(self, parameters: dict = {}):
        self.fromDict(parameters)

    def fromDict(self, parameters: dict, update=False):
        self.vert_indices = BasePrimitiveSurface.readParameterOnDict('vert_indices', parameters, old_value=(self.vert_indices if update else None))
        self.vert_parameters = BasePrimitiveSurface.readParameterOnDict('vert_parameters', parameters, old_value=(self.vert_parameters if update else None))
        self.face_indices = BasePrimitiveSurface.readParameterOnDict('face_indices', parameters, old_value=(self.face_indices if update else None))
    
    def toDict(self):
        parameters = {}
        BasePrimitiveSurface.writeParameterOnDict('vert_indices', self.vert_indices, parameters)
        BasePrimitiveSurface.writeParameterOnDict('vert_parameters', self.vert_parameters, parameters)
        BasePrimitiveSurface.writeParameterOnDict('face_indices', self.face_indices, parameters)
        return parameters

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
