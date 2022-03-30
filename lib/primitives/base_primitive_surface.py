from abc import abstractmethod
from lib.utils import angleVectors
import numpy as np

# I = 0
# error = []
# error_T = []
# not_error = []
# not_error_T = []

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
        # global I
        points_normals = np.array([self._computeCorrectPointAndNormal(P) for P in points], dtype=points.dtype)
        # if self.getPrimitiveType() == 'cone':
        #     points2 = points_normals[:, :3]
        #     size1 = np.max(points2, axis=0) - np.min(points2, axis=0)
        #     size1 = size1[0]*size1[1]*size1[2]
        #     size2 = np.max(points2, axis=0) - np.min(points2, axis=0)
        #     size2 = size2[0]*size2[1]*size2[2]
        #     if len(points) >= 3000:
        #         if size2 == 0 or size2/size1 <= 0.5:
        #             print(f'Cone {I}, ERROR:')
        #             print(self.toDict())
        #         else:
        #             print(f'Cone {I}, RIGHT:')
        #             print(self.toDict())

        #     I += 1

        return points_normals[:, :3], points_normals[:, 3:]

    def computeErrors(self, points, normals):
        assert len(points) > 0 and len(normals) > 0
        new_points, new_normals = self.computeCorrectPointsAndNormals(points)
        distances = np.array([np.linalg.norm(P - points[i], ord=2) for i, P in enumerate(new_points)], dtype=new_points.dtype)
        angles = np.array([angleVectors(n, normals[i]) for i, n in enumerate(new_normals)], dtype=new_normals.dtype)
        result = {'distances': distances, 'angles': angles}
        return result
