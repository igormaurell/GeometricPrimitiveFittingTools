from abc import abstractmethod
import numpy as np

class PrimitiveSurface:
    @staticmethod
    def readParameterOnDict(key, d, old_value=None):
        return None if key not in d.keys and old_value is None else d[key]
    
    @staticmethod
    def writeParameterOnDict(key, value, d):
        if value is not None:
            d[key] = value

    def __init__(self):
        self.vert_indices = None
        self.vert_parameters = None
        self.face_indices = None

    def fromDict(self, parameters: dict, update=False):
        self.vert_indices = PrimitiveSurface.readParameterOnDict('vert_indices', parameters, old_value=(self.vert_indices if update else None))
        self.vert_parameters = PrimitiveSurface.readParameterOnDict('vert_parameters', parameters, old_value=(self.vert_parameters if update else None))
        self.face_indices = PrimitiveSurface.readParameterOnDict('face_indices', parameters, old_value=(self.face_indices if update else None))
    
    def toDict(self):
        parameters = {}
        PrimitiveSurface.writeParameterOnDict('vert_indices', self.vert_indices, parameters)
        PrimitiveSurface.writeParameterOnDict('vert_parameters', self.vert_parameters, parameters)
        PrimitiveSurface.writeParameterOnDict('face_indices', self.face_indices, parameters)
        return parameters

    def computeErrors(self, points, normals, deviation_function):
        assert len(points) == len(normals)
        distances = np.zeros(len(points))
        angles = np.zeros(len(points))
        for i in range(len(points)):
            distance, angle = deviation_function(points[i], normals[i], self.location, self.z_axis, self.radius)
            distances[i] = distance
            angles[i] = angle
        result = {}
        result['mean_distance_error'] = np.mean(distances)
        result['mean_angle_error'] = np.mean(angles)
        result['distance_errors'] = distances
        result['angle_errors'] = angles
        return result