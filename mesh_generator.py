import argparse
from os.path import join
from pypcd import pypcd
import numpy as np
from scipy.spatial import Delaunay

import matplotlib.pyplot as plt

from scipy.spatial import Delaunay

import open3d as o3d
import json

from tqdm.contrib.concurrent import thread_map

from functools import partial

from lib.surfaces_projector import SurfacesProjector

import os
from pathlib import Path

import igl

import json

EPS = np.finfo(float).eps
LEAF_SIZE = 0.02
N_NEIGHBORS = 10
MAX_WORKERS = 8
GENERATE_TRIANGULATION_FEATURES = False
PROJECT_POINTS = False
FILTER = True

def computeLabelsFromFace2Primitive(labels, features_data):
    max_face = np.max(labels)
    for feat in features_data:
        max_face = max(0 if len(feat['face_indices']) == 0 else max(feat['face_indices']), max_face)
    face_2_primitive = np.zeros(shape=(max_face+1,), dtype=np.int32) - 1
    face_primitive_count = np.zeros(shape=(max_face+1,), dtype=np.int32)
    for i, feat in enumerate(features_data):
        for face in feat['face_indices']:
            face_2_primitive[face] = i
            face_primitive_count[face] += 1
    assert len(np.unique(face_primitive_count)) <= 2
    features_point_indices = [[] for i in range(0, len(features_data) + 1)]
    for i in range(0, len(labels)):
        index = face_2_primitive[labels[i]]
        features_point_indices[index].append(i)
        labels[i] = index
    features_point_indices.pop(-1)

    for i in range(0, len(features_point_indices)):
        features_point_indices[i] = np.array(features_point_indices[i], dtype=np.int64)
    
    return labels, features_point_indices

def compute_local_densities(pcd, k=N_NEIGHBORS):
    if len(pcd.points) > 1:
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        points = np.asarray(pcd.points)
        distances = []
        for i in range(len(points)):
            A = points[i]
            _, idx, _ = pcd_tree.search_knn_vector_3d(A, k)
            points_query = points[np.asarray(idx)]
            B = points_query[-1]
            dist = np.linalg.norm(B-A, ord=2)
            distances.append(dist)
        dist_arr = np.array(distances)
        return np.min(dist_arr, axis=0), np.mean(dist_arr, axis=0), np.max(dist_arr, axis=0)
    else:
        return 0, 0, 0

def filter_pcd(pcd):
    if FILTER:
        pcd = pcd.voxel_down_sample(voxel_size=LEAF_SIZE)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=N_NEIGHBORS, std_ratio=2.0)
    return pcd

def delaunay_uv_triangulation(pcd, surface):
    pcd = filter_pcd(pcd)
    if len(pcd.points) > 2:
        _, _, max_r = compute_local_densities(pcd) 

        points = np.asarray(pcd.points)

        response = SurfacesProjector.projectPointsOnSurfaceFeatures(points, surface)

        if response is None:
            return o3d.geometry.TriangleMesh()
        
        uvs, _ = response

        tri = Delaunay(uvs)
        triangles = tri.simplices

        triangles_filter = []
        index_perm = [(0,1), (0,2), (1,2)]
        for t in triangles:
            keep = True
            for i, j in index_perm:
                A = points[t[i]]
                B = points[t[j]]
                dist = np.linalg.norm(B - A, ord=2)
                if dist > max_r:
                    keep = False
                    break
            triangles_filter.append(keep)
        triangles_filter = np.array(triangles_filter)

        triangles = triangles[triangles_filter]

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(points)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)

        plt.scatter(uvs[:, 0], uvs[:, 1])
        plt.show()

        return mesh
    else:
        return o3d.geometry.TriangleMesh()

def bpa_triangulation(pcd, surface=None):
    pcd = filter_pcd(pcd)
    if len(pcd.points) > 2:
        _, _, max_r = compute_local_densities(pcd)
        radii = [max_r/2, max_r]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii))
        return mesh
    else:
        return o3d.geometry.TriangleMesh()
    
def poisson_triangulation(pcd, surface=None):
    pcd = filter_pcd(pcd)
    if len(pcd.points) > 2:
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9)
        return mesh
    else:
        return o3d.geometry.TriangleMesh()

def triangulation(pcd, data):
    pcd_surf = pcd.select_by_index(data['point_indices'])

    mesh = o3d.geometry.TriangleMesh()

    if len(pcd_surf.points) > 0:
        result = SurfacesProjector.projectPointsOnSurfaceFeatures(np.asarray(pcd_surf.points), data)
        if result is not None:
            uvs, points = result
            data['point_parameters'] = uvs
            if PROJECT_POINTS:
                pcd_surf.points = o3d.utility.Vector3dVector(points)

        mesh = bpa_triangulation(pcd_surf, surface=data)

    return mesh

def triangulation_by_surface(pcd, surfaces_data):
    func = partial(triangulation, pcd)
    result = thread_map(func, surfaces_data, max_workers=MAX_WORKERS, chunksize=1)
        
    final_mesh = o3d.geometry.TriangleMesh()
    for i, r in enumerate(result):
        surfaces_data[i]['vert_indices'] = list(range(len(final_mesh.vertices), len(final_mesh.vertices) + len(r.vertices)))
        del surfaces_data[i]['vert_parameters'] #missing
        res = SurfacesProjector.projectPointsOnSurfaceFeatures(np.asarray(r.vertices), surfaces_data[i])
        if res is not None:
            uvs, _ = res
            surfaces_data[i]['vert_parameters'] = uvs if type(uvs) is list else uvs.tolist()
        surfaces_data[i]['face_indices'] = list(range(len(final_mesh.triangles), len(final_mesh.triangles) + len(r.triangles)))
        surfaces_data[i]['point_indices'] = surfaces_data[i]['point_indices'] if type(surfaces_data[i]['point_indices']) is list else surfaces_data[i]['point_indices'].tolist()
        if 'point_parameters' in surfaces_data[i]:
            surfaces_data[i]['point_parameters'] = surfaces_data[i]['point_parameters'] if type(surfaces_data[i]['point_parameters']) is list else surfaces_data[i]['point_parameters'].tolist()

        final_mesh += r
        
    final_mesh.compute_triangle_normals()
    return final_mesh

def list_files(input_dir: str, return_str=False) -> list:
    files = []
    path = Path(input_dir)
    for file_path in path.glob('*'):
        files.append(file_path if not return_str else str(file_path))
    return sorted(files)

def list_filenames(input_dir):
    files = list_files(input_dir, return_str=True)
    filenames = []
    for f in files:
        start_index = f.rfind('/')
        start_index = start_index if start_index != -1 else 0
        final_index = f.rfind('.')
        filenames.append(f[(start_index+1):final_index])
    return filenames


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('folder', type=str, help='')
    parser.add_argument('-gf','--generate_triangulation_features', action='store_true', help='')
    parser.add_argument('-pp','--project_points', action='store_true', help='')
    parser.add_argument('-r','--reegenerate', action='store_true', help='')
    parser.add_argument('-nf','--no_filter', action='store_true', help='')
    parser.add_argument('--filename', type=str, default='', help='')
    parser.add_argument('--pc_foldername', type=str, default='pc', help='')
    parser.add_argument('--triangulation_foldername', type=str, default='triangulation', help='')
    parser.add_argument('--triangulation_features_foldername', type=str, default='triangulation_features', help='')
    parser.add_argument('--n_neighbors', type=int, default=N_NEIGHBORS, help='')
    parser.add_argument('--leaf_size', type=float, default=LEAF_SIZE, help='')
    parser.add_argument('--max_workers', type=int, default=MAX_WORKERS, help='')


    args = vars(parser.parse_args())

    folder_name = args['folder']
    filename = args['filename']
    pc_foldername = args['pc_foldername']
    triangulation_foldername = args['triangulation_foldername']
    triangulation_features_foldername = args['triangulation_features_foldername']
    GENERATE_TRIANGULATION_FEATURES = args['generate_triangulation_features']
    PROJECT_POINTS = args['project_points']
    N_NEIGHBORS = args['n_neighbors']
    LEAF_SIZE = args['leaf_size']
    MAX_WORKERS = args['max_workers']
    FILTER = not args['no_filter']
    reegenerate = args['reegenerate']

    if not os.path.exists(folder_name) or not os.path.isdir(folder_name):
        print('[Generator Error] Input path not found')
        exit()

    filenames = []
    if filename != '':
        filenames.append(filename)
    else:
        filenames = list_filenames(os.path.join(folder_name, 'features'))

    if not reegenerate:
        filenames_triang = list_filenames(os.path.join(folder_name, triangulation_foldername))
        filenames_triang_features = list_filenames(os.path.join(folder_name, triangulation_features_foldername))


        diff_1 = set(filenames) - set(filenames_triang)
        diff_2 = set(filenames) - set(filenames_triang_features)

        if GENERATE_TRIANGULATION_FEATURES:
            diff_1 = diff_1.union(diff_2)
        
        filenames = list(diff_1)
    
    os.makedirs(os.path.join(folder_name, triangulation_foldername), exist_ok=True)

    if GENERATE_TRIANGULATION_FEATURES:
        os.makedirs(os.path.join(folder_name, triangulation_features_foldername), exist_ok=True)

    for filename in filenames:
        print()
        print('Processing file {}:'.format(filename))

        pc_filename = join(folder_name, 'pc', filename + '.pcd')

        geometry_filename = join(folder_name, 'features', filename + '.json')

        pc = pypcd.PointCloud.from_path(pc_filename).pc_data

        pointcloud = o3d.geometry.PointCloud()

        if 'x' in pc.dtype.names and 'y' in pc.dtype.names and 'z' in pc.dtype.names:
            pointcloud.points = o3d.utility.Vector3dVector(np.vstack((pc['x'], pc['y'], pc['z'])).T)
        else:
            print('there is no point in the input pointcloud')
            exit()

        if 'normal_x' in pc.dtype.names and 'normal_y' in pc.dtype.names and 'normal_z' in pc.dtype.names:
            pointcloud.normals = o3d.utility.Vector3dVector(np.vstack((pc['normal_x'], pc['normal_y'], pc['normal_z'])).T)
        else:
            print('there is no normal in the input pointcloud')

        has_labels = False

        if 'label' in pc.dtype.names:
            labels = pc['label']
            has_labels = True
        else:
            print('there is no label in the input pointcloud')
        

        with open(geometry_filename, 'r') as f:
            geometry_data = json.load(f)
            surfaces_data = geometry_data['surfaces']

        if has_labels:
            labels, fpi = computeLabelsFromFace2Primitive(labels, surfaces_data)
            for i in range(len(fpi)):
                surfaces_data[i]['point_indices'] = fpi[i]
        
        if not pointcloud.has_normals():
            pointcloud.estimate_normals()

        mesh = triangulation_by_surface(pointcloud, surfaces_data)
        igl.write_triangle_mesh(os.path.join(folder_name, triangulation_foldername, filename + '.obj'), np.asarray(mesh.vertices), np.asarray(mesh.triangles))

        if GENERATE_TRIANGULATION_FEATURES:
            features = {'surfaces': surfaces_data}
            with open(os.path.join(folder_name, triangulation_features_foldername, filename + '.json'), 'w') as f:
                json.dump(features, f, indent=4)