import pickle
import h5py
import csv
import numpy as np
import gc
import uuid
import os
from sklearn.neighbors import KDTree
from collections import Counter
import tqdm
import shutil

from lib.normalization import normalize
from lib.utils import filterFeaturesData, translateFeature, computeLabelsFromFace2Primitive, computeFeaturesPointIndices, strLower

from .base_dataset_writer import BaseDatasetWriter

class PrimitivenetDatasetWriter(BaseDatasetWriter):
    FEATURES_BY_TYPE = {
        'plane': ['type', 'name', 'location_x', 'location_y', 'location_z', 'axis_x', 'axis_y', 'axis_z', 'normalized'],
        'cylinder': ['type', 'name', 'location_x', 'location_y', 'location_z', 'axis_x', 'axis_y', 'axis_z', 'radius', 'normalized'],
        'cone': ['type', 'name', 'location_x', 'location_y', 'location_z', 'axis_x', 'axis_y', 'axis_z', 'radius', 'semi_angle', 'apex_x', 'apex_y', 'apex_z', 'normalized'],
        'sphere': ['type', 'name', 'location_x', 'location_y', 'location_z', 'radius', 'normalized']
    }

    FEATURES_MAPPING = {
        'type': {'type': str, 'map': 'type', 'transform': strLower},
        'name': {'type': str, 'map': 'name'},
        'normalized': {'type': bool, 'map': 'normalized'},
        'location_x': {'type': float, 'map': ('location', 0)},
        'location_y': {'type': float, 'map': ('location', 1)},
        'location_z': {'type': float, 'map': ('location', 2)},
        'axis_x': {'type': float, 'map': ('z_axis', 0)},
        'axis_y': {'type': float, 'map': ('z_axis', 1)},
        'axis_z': {'type': float, 'map': ('z_axis', 2)},
        'apex_x': {'type': float, 'map': ('apex', 0)},
        'apex_y': {'type': float, 'map': ('apex', 1)},
        'apex_z': {'type': float, 'map': ('apex', 2)},
        'semi_angle': {'type': float, 'map': 'angle'},
        'radius': {'type': float, 'map': 'radius'},
    }


    def __init__(self, parameters):
        super().__init__(parameters)
        type_list = list(set(PrimitivenetDatasetWriter.FEATURES_BY_TYPE.keys()))
        self.type2idx = dict()
        for i, k in enumerate(type_list):
            self.type2idx[PrimitivenetDatasetWriter.FEATURES_MAPPING['type']['transform'](k)] = i + 1

    def step(self, points, normals=None, labels=None, features_data=[], noisy_points=None, filename=None, features_point_indices=None, mesh=None):
        if filename is None:
            filename = str(uuid.uuid4())
        
        data_file_path = os.path.join(self.data_folder_name, f'{filename}.npz')
        transforms_file_path = os.path.join(self.transform_folder_name, f'{filename}.pkl')

        vertices = mesh[0]
        faces = mesh[1]

        normals = np.zeros(shape=(vertices.shape[0],3), dtype=np.float)

        if os.path.exists(data_file_path):
           return False

        labels = np.zeros(len(vertices), dtype=np.int) - 1

        for i, feature in enumerate(features_data['surfaces']):
            labels[np.array(feature['vert_indices'], dtype=np.int)] = i

        normals_face = np.zeros(faces.shape, dtype=np.float64)
        centers_face = np.zeros(faces.shape, dtype=np.float64)
        for i, f in enumerate(faces):
            A = vertices[f[0]]
            B = vertices[f[1]]
            C = vertices[f[2]]
            v1 = A - C
            v2 = B - C
            n = np.cross(v1, v2)
            normals_face[i] = n/(np.linalg.norm(n, ord=2) + np.finfo(float).eps)
            center = (A + B + C)/3
            centers_face[i] = center
                
        features_vertice_indices = computeFeaturesPointIndices(labels)


        min_number_points = self.min_number_points if self.min_number_points > 1 else int(len(labels)*self.min_number_points)
        min_number_points = min_number_points if min_number_points >= 0 else 1

        features_data['surfaces'], labels, features_vertice_indices = filterFeaturesData(features_data['surfaces'], types=self.filter_features_parameters['surface_types'], min_number_points=min_number_points,
                                                                 labels=labels, features_point_indices=features_vertice_indices)

        if len(features_data['surfaces']) == 0:
            print(f'ERROR: {data_file_path} has no features left.')
            return False

        labels_faces = np.zeros(len(faces), dtype=np.int) - 1
        types_faces = np.ones(len(faces), dtype=np.int)

        for i, feature in enumerate(features_data['surfaces']):
            labels_faces[np.array(feature['face_indices'], dtype=np.int)] = i
            types_faces[np.array(feature['face_indices'], dtype=np.int)] = self.type2idx[PrimitivenetDatasetWriter.FEATURES_MAPPING['type']['transform'](feature['type'])]
        
        #vert_instance_counts = np.zeros((len(vertices), len(labels_faces)), dtype=np.int)

        #for face_idx, instance_idx in enumerate(labels_faces):
        #    for vertice_idx in faces[face_idx]:
        #        vert_instance_counts[vertice_idx, instance_idx] += 1

        boundaries = np.zeros(len(vertices), dtype=np.bool)
        #for vertice_idx, instance_counts in enumerate(vert_instance_counts):
        #    if instance_counts.any() > 1:
        #        boundaries[vertice_idx] = True

        '''for vertice_idx, vertice in enumerate(vertices):
            vertice_instances = labels_faces[(faces[...,0] == vertice_idx) + (faces[...,1] == vertice_idx) + (faces[...,2] == vertice_idx)].reshape((-1,))
            boundaries[vertice_idx] = len(set(vertice_instances)) > 1'''

        for i, feature in enumerate(features_data['curves']):
            boundaries[np.array(feature['vert_indices'], dtype=np.int)] = True

        vert_face_counts = np.zeros((len(vertices),1), dtype=np.int)
        for face_idx, face in enumerate(faces):
            vert_face_counts[face,0] += 1
            normals[face] += normals_face[face_idx]

        normals = normals / vert_face_counts

        self.filenames_by_set[self.current_set_name].append(filename)

        noise_limit = 0.

        if 'add_noise' in self.normalization_parameters.keys():
            noise_limit = self.normalization_parameters['add_noise']
            self.normalization_parameters['add_noise'] = 0.
                
        #print(vertices.shape, normals.shape)
        vertices, normals, features_data['surfaces'], transforms = normalize(vertices, self.normalization_parameters, normals=normals.copy(), features=features_data['surfaces'])
        #print(vertices.shape, normals.shape)
        with open(transforms_file_path, 'wb') as pkl_file:
            pickle.dump(transforms, pkl_file)

        npz_fields = {
            'V': vertices,
            'N': normals,
            'I': labels_faces,
            'VC': centers_face,
            'B': boundaries,
            'F': faces,
            'FN': normals_face,
            'S': types_faces,
        }
        
        np.savez(data_file_path, **npz_fields)
                   
        return True

    def finish(self, permutation=None):
        train_models, test_models = self.divisionTrainVal(permutation=permutation)

        os.makedirs(os.path.join(self.data_folder_name, 'train'), exist_ok=True)

        for filename in train_models:
            data_source_path = os.path.join(self.data_folder_name, f'{filename}.npz')
            data_target_path = os.path.join(self.data_folder_name, 'train', f'{filename}.npz')
            shutil.move(data_source_path, data_target_path)

        os.makedirs(os.path.join(self.data_folder_name, 'val'), exist_ok=True)

        for filename in test_models:
            data_source_path = os.path.join(self.data_folder_name, f'{filename}.npz')
            data_target_path = os.path.join(self.data_folder_name, 'val', f'{filename}.npz')
            shutil.move(data_source_path, data_target_path)

        super().finish()