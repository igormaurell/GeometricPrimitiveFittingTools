import os
import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay
from functools import partial
from tqdm.contrib.concurrent import thread_map
from pypcd import pypcd

from .surfaces_projector import SurfacesProjector

# Constants
N_NEIGHBORS = 10
FILTER = True       # Apply voxel grid and remove statistical outlier?
LEAF_SIZE = 0.02    # Voxel size
PROJECT_POINTS = False
MAX_WORKERS = 8

def compute_labels_from_face_2_primitive(labels: list, features_data: dict):
    max_face = np.max(labels)
    for feature in features_data:
        max_face = max(0 if len(feature["face_indices"]) == 0 else max(feat["face_indices"]), max_face)
    face_2_primitive = np.zeros(shape=(max_face+1,), dtype=np.int32) - 1
    face_primitive_count = np.zeros(shape=(max_face+1,), dtype=np.int32)
    for feature_idx, feature in enumerate(features_data):
        for face_id in feature["face_indices"]:
            face_2_primitive[face_id] = feature_idx
            face_primitive_count[face_id] += 1
    assert len(np.unique(face_primitive_count)) <= 2
    
    features_point_indices = [[] for i in range(0, len(features_data)+1)]
    for i in range(0, len(labels)):
        index = face_2_primitive[labels[i]]
        features_point_indices[index].append(i)
        labels[i] = index
    features_point_indices.pop(-1)

    for i in range(0, len(features_point_indices)):
        features_point_indices[i] = np.array(features_point_indices[i], dtype=np.int64)

    return labels, features_point_indices

def compute_local_densities(pcd: o3d.geometry.PointCloud, k: int = N_NEIGHBORS):
    if len(pcd.points) <= 1:
        return 0, 0, 0
    
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    points = np.asarray(pcd.points)
    distances = []
    for i in range(len(points)):
        A = points[i]
        _, idx, _ = pcd_tree.search_knn_vector_3d(A, k)
        points_query = points[np.asarray(idx)]
        B = points_query[-1]
        distances.append(np.linalg.norm(B-A, ord=2))
    dist_arr = np.array(distances)
    return np.min(dist_arr, axis=0), np.mean(dist_arr, axis=0), np.max(dist_arr, axis=0)
    

def filter_pcd(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    if FILTER:
        pcd = pcd.voxel_down_sample(voxel_size=LEAF_SIZE)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=N_NEIGHBORS, std_ratio=2.0)
    return pcd

# def delaunay_uv_triangulation(pcd: o3d.geometry.PointCloud, surface) -> o3d.geometry.TriangleMesh:
#     pcd = filter_pcd(pcd)
#     if len(pcd.points) <= 2:
#         return o3d.geometry.TriangleMesh()
    
#     _, _, max_r = compute_local_densities(pcd)

#     points = np.asarray(pcd.points)

#     response = SurfacesProjector.projectPointsOnSurfaceFeatures(points, surface)

#     if response is None:
#         return o3d.geometry.TriangleMesh()
    
#     uvs, _ = response

#     tri = Delaunay(uvs)
#     triangles = tri.simplices

#     triangles_filter = []
#     index_perm = [(0,1), (0,2), (1,2)]
#     for triangle in triangles:
#         keep = True
#         for i, j in index_perm:
#             A = points[triangle[i]]
#             B = points[triangle[j]]
#             dist = np.linalg.norm(B-A, ord=2)
#             if dist > max_r:
#                 keep = False
#                 break
#         triangles_filter.append(keep)
#     triangles_filter = np.array(triangles_filter)

#     triangles = triangles[triangles_filter]

#     mesh = o3d.geometry.TriangleMesh()
#     mesh.vertices = o3d.utility.Vector3dVector(points)
#     mesh.triangles = o3d.utility.Vector3iVector(triangles)

#     plt.scatter(uvs[:,0], uvs[:,1])
#     plt.show()

#     return mesh

def bpa_triangulation(pcd: o3d.geometry.PointCloud, surface = None) -> o3d.geometry.TriangleMesh:
    pcd = filter_pcd(pcd)
    if len(pcd.points) <= 2:
        return o3d.geometry.TriangleMesh()
    
    _, _, max_r = compute_local_densities(pcd)
    radii = [max_r/2, max_r]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    return mesh

def poisson_triangulation(pcd: o3d.geometry.PointCloud, surface = None) -> o3d.geometry.TriangleMesh:
    pcd = filter_pcd(pcd)
    if len(pcd.points) <= 2:
        return o3d.geometry.TriangleMesh()
    
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9
    )
    return mesh

def triangulation(pcd: o3d.geometry.PointCloud, data: dict) -> o3d.geometry.TriangleMesh:
    pcd_surf = pcd.select_by_index(data["point_indices"])

    mesh = o3d.geometry.TriangleMesh()

    if len(pcd_surf.points) > 0:
        result = SurfacesProjector.projectPointsOnSurfaceFeatures(np.asarray(pcd_surf.points), data)
        if result is not None:
            uvs, points = result
            data["point_parameters"] = uvs
            if PROJECT_POINTS:
                pcd_surf.points = o3d.utility.Vector3dVector(points)
        mesh = bpa_triangulation(pcd_surf, surface=data)
    
    return mesh

def triangulation_by_surface(pcd: o3d.geometry.PointCloud, surfaces_data: dict) -> o3d.geometry.TriangleMesh:
    func = partial(triangulation, pcd)
    result_ = thread_map(func, surfaces_data, max_workers=MAX_WORKERS, chunk_size=1)

    final_mesh = o3d.geometry.TriangleMesh()
    for result_idx, result in enumerate(result_):
        surfaces_data[result_idx]["vert_indices"] = list(range(len(final_mesh.vertices),
                                                               len(final_mesh.vertices)+len(result.vertices)))
        del surfaces_data[result_idx]["vert_parameters"]
        res = SurfacesProjector.projectPointsOnSurfaceFeatures(np.asarray(result.vertices),
                                                               surfaces_data[result_idx])
        if res is not None:
            uvs, _ = res
            surfaces_data[result_idx]["vert_parameters"] = uvs if type(uvs) is list else uvs.tolist()
        surfaces_data[result_idx]["face_indices"] = list(range(len(final_mesh.triangles),
                                                               len(final_mesh.trangles)+len(result.triangles)))
        surfaces_data[result_idx]["point_indices"] = surfaces_data[result_idx]["point_indices"] \
                                                        if type(surfaces_data[result_idx]["point_indices"]) is list \
                                                            else surfaces_data[i]['point_indices'].tolist()
        if "point_parameters" in surfaces_data[result_idx]:
            surfaces_data[result_idx]["point_parameters"] = surfaces_data[result_idx]["point_parameters"] \
                                                                if type(surfaces_data[result_idx]["point_parameters"]) is list \
                                                                    else surfaces_data[result_idx]["point_parameters"].tolist()

        final_mesh += result
    final_mesh.compute_triangle_normals()
    return final_mesh

def generate_mesh_from_pointcloud(args, pc: pypcd.PointCloud, pc_filename: str, geometry_data: dict) -> bool:
    filename = pc_filename
    foldername = args['mesh_foldername']
    triangulation_foldername = args['triangulation_foldername']
    triangulation_features_foldername = args['triangulation_features_foldername']
    GENERATE_TRIANGULATION_FEATURES = args['generate_triangulation_features']
    PROJECT_POINTS = args['project_points']
    N_NEIGHBORS = args['n_neighbors_mesh']
    LEAF_SIZE = args['leaf_size_mesh']
    MAX_WORKERS = args['max_workers_mesh']
    FILTER = not args['no_filter']
    reegenerate = args['reegenerate']

    print()
    print("Generating mesh for the file: {}".format(filename))

    pointcloud = o3d.geometry.PointCloud()

    if 'x' in pc.dtype.names and 'y' in pc.dtype.names and 'z' in pc.dtype.names:
        pointcloud.points = o3d.utility.Vector3dVector(np.vstack((pc['x'], pc['y'], pc['z'])).T)
    else:
        print("There is no point in the input pointcloud")
        return False
    
    if 'normal_x' in pc.dtype.names and 'normal_y' in pc.dtype.names and 'normal_z' in pc.dtype.names:
        pointcloud.normals = o3d.utility.Vector3dVector(np.vstack((pc['normal_x'], pc['normal_y'], pc['normal_z'])).T)
    else:
        print('There is no normal in the input pointcloud')
    
    has_labels = False
    if 'label' in pc.dtype.names:
        labels = pc['label']
        has_labels = True
    else:
        print("there is no label in the input pointcloud")

    surfaces_data = geometry_data['surfaces']

    if has_labels:
        labels, fpi = compute_labels_from_face_2_primitive(labels, surfaces_data)
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
    
    return True
