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
            self.type2idx[PrimitivenetDatasetWriter.FEATURES_MAPPING['type']['transform'](k)] = i
        self.type2idx['unlabeled'] = len(type_list)

    def step(self, points, normals=None, labels=None, features_data=[], noisy_points=None, filename=None, features_point_indices=None, mesh=None):
        if filename is None:
            filename = str(uuid.uuid4())
        
        data_file_path = os.path.join(self.data_folder_name, f'{filename}.npz')
        transforms_file_path = os.path.join(self.transform_folder_name, f'{filename}.pkl')

        if type(features_data) == dict:
            features_data = features_data['surfaces']

        if os.path.exists(data_file_path):
           return False

        if labels is not None:   
            if features_point_indices is None:
                features_point_indices = computeFeaturesPointIndices(labels)

            min_number_points = self.min_number_points if self.min_number_points > 1 else int(len(labels)*self.min_number_points)
            min_number_points = min_number_points if min_number_points >= 0 else 1

            features_data, labels, features_point_indices = filterFeaturesData(features_data, types=self.filter_features_parameters['surface_types'], min_number_points=min_number_points,
                                                           labels=labels, features_point_indices=features_point_indices)

            if len(features_data) == 0:
                print(f'ERROR: {data_file_path} has no features left.')
                return False


        self.filenames_by_set[self.current_set_name].append(filename)

        noise_limit = 0.

        if 'add_noise' in self.normalization_parameters.keys():
            noise_limit = self.normalization_parameters['add_noise']
            self.normalization_parameters['add_noise'] = 0.
                
        points, normals, features_data, transforms = normalize(points, self.normalization_parameters, normals=normals.copy(), features=features_data)

        #vertices = points
        #vertex_normals = normals

        with open(transforms_file_path, 'wb') as pkl_file:
            pickle.dump(transforms, pkl_file)

        '''F = []

        for surface in features_data['surfaces']:
            surface['face_indices']
            surface['type']
            surface['vert_indices']'''

        '''cloud = open3d.geometry.PointCloud(open3d.cuda.pybind.utility.Vector3dVector(points))
        cloud.estimate_normals()
        mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud)[0]
        mesh.compute_vertex_normals()

        vertices = np.array(mesh.vertices)
        vertex_normals = np.array(mesh.vertex_normals)
        triangles = np.array(mesh.triangles)'''



        '''tree = KDTree(vertices, leaf_size=256)
        print('Done building KDTree.')
        _, triangles = tree.query(vertices, k=3)
        print('Done querying KDTree.')
        triangles = np.unique(np.sort(triangles), axis=0)
        print('Done findind unique triangles.')'''

        #triangles = mesh.faces

        #face_types = np.ones(shape=triangles.shape[:1], dtype=np.int32) * self.type2idx['unlabeled']
        surfaces_vertex_indices = []

        point_types = np.ones(shape=(points.shape[0],1), dtype=np.int32) * (len(self.type2idx.keys()) - 1)

        for surface in tqdm.tqdm(features_data):
            if PrimitivenetDatasetWriter.FEATURES_MAPPING['type']['transform'](surface['type']) not in self.type2idx:
                continue
            #face_types[surface_indices] = self.type2idx[PrimitivenetDatasetWriter.FEATURES_MAPPING['type']['transform'](surface['type'])]
            #surface_faces = triangles[surface_indices]
            point_types[surface['vert_indices'],0] = self.type2idx[PrimitivenetDatasetWriter.FEATURES_MAPPING['type']['transform'](surface['type'])]

            

        '''vertex_instances = np.ones(shape=(vertices.shape[0],), dtype=np.int32) * -1
        for i, (surface, vert_index) in tqdm.tqdm(enumerate(zip(surfaces_data, surfaces_vertex_indices))):
            vertex_instances[vert_index] = i'''
        
        npz_fields = {
            'V': points,
            'N': normals,
            'I': labels,
            'B': np.zeros(shape=(points.shape[0],), dtype=bool),
            #'F': triangles,
            'S': point_types,
        }
        

        '''for idx, face in tqdm.tqdm(enumerate(triangles)):
            v0_instance = vertex_instances[face[0]]
            v1_instance = vertex_instances[face[1]]
            v2_instance = vertex_instances[face[2]]
            instance_counter = Counter([v0_instance,v1_instance,v2_instance])
            for v_instance, v_idx in zip((v0_instance, v1_instance, v2_instance), face):
                if instance_counter[v_instance] == 1:
                    npz_fields['B'][v_idx] = 1'''


        '''types_mesh = mesh.copy()
        type_colors = np.random.sample(size=())'''
        #for curve in tqdm.tqdm(curves_data):
        #    npz_fields['B'][curve['vert_indices']] = np.ones(shape=(len(curve['vert_indices']),), dtype=np.int32)

        #del normals

        #gc.collect()

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