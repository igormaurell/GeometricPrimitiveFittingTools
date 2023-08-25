from collections.abc import Iterable
import numpy as np
import pickle
import json
import yaml
from numba import njit
from os.path import exists
from os import system
from pypcd import pypcd
import open3d as o3d
from scipy.optimize import least_squares
from tqdm import tqdm
import random
import colorsys
from typing import List, Tuple
import functools
from copy import deepcopy

EPS = np.finfo(np.float64).eps

def funif(fun, var: bool):
    return fun if var else lambda x: x

@functools.lru_cache(1000)
def get_evenly_distributed_colors(count: int) -> List[Tuple[np.uint8, np.uint8, np.uint8]]:
    # lru cache caches color tuples
    HSV_tuples = [(x/count, 1.0, 1.0) for x in range(count)]
    random.shuffle(HSV_tuples)
    return list(map(lambda x: (np.array(colorsys.hsv_to_rgb(*x))*255).astype(np.uint8), HSV_tuples))

@njit
def sortedIndicesIntersection(a, b):
    i = 0
    j = 0
    k = 0
    intersect = np.empty_like(a)
    while i< a.size and j < b.size:
            if a[i]==b[j]: # the 99% case
                intersect[k]=a[i]
                k+=1
                i+=1
                j+=1
            elif a[i]<b[j]:
                i+=1
            else : 
                j+=1
    return intersect[:k]

'''COLOR'''
def computeRGB(value):
    r = value%256
    value = value//256
    g = value%256
    b = value//256
    return (r, g, b)

def getAllColorsArray():
    np.random.seed(0)
    colors = np.random.permutation(256*256*256)
    return colors

'''POINT CLOUDS'''

def generatePCD(pc_filename, mps_ns, mesh_filename=None): 
    if not exists(pc_filename):
        if mesh_filename is None:
            return []
        system(f'mesh_point_sampling {mesh_filename} {pc_filename} --n_samples {mps_ns} --write_normals --no_vis_result > /dev/null')
    return True

def writeColorPointCloudOBJ(out_filename, point_cloud):
    with open(out_filename, 'w') as fout:
        text = ''
        for point in point_cloud:
            text += 'v %f %f %f %d %d %d\n' % (point[0], point[1], point[2], point[3], point[4], point[5])
        fout.write(text)

'''FEATURES'''

YAML_NAMES = ['yaml', 'yml']
JSON_NAMES = ['json']
PKL_NAMES  = ['pkl']

# Load features file
def loadYAML(features_name: str):
    with open(features_name, 'r') as f:
        data = yaml.load(f, Loader=yaml.Loader)
    return data

def loadJSON(features_name: str):
    with open(features_name, 'r') as f:
        data = json.load(f)
    return data

def loadPKL(features_name: str):
    with open(features_name, 'rb') as f:
        data = pickle.load(f)
    return data

def loadFeatures(features_name: str, tp: str):
    if tp.lower() in YAML_NAMES:
        return loadYAML(f'{features_name}.{tp}')
    elif tp.lower() in PKL_NAMES:
        return loadPKL(f'{features_name}.{tp}')
    else:
        return loadJSON(f'{features_name}.{tp}')

def strUpperFirstLetter(s):
    return s[0].upper() + s[1:]

def strLower(s):
    return s.lower()

def translateFeature(feature_data, features_by_type, features_mapping):
    assert 'type' in feature_data.keys()
    tp = feature_data['type'].lower()
    assert tp in features_by_type.keys()
    feature_out = {}
    for feature_key in features_by_type[tp]:
        if feature_key in features_mapping:
            mapping = features_mapping[feature_key]
            feature_out[feature_key] = mapping['type']()
            if isinstance(mapping['map'], list):
                for elem in mapping['map']:
                    if isinstance(elem, tuple):
                        feature_out[feature_key].append(feature_data[elem[0]][elem[1]])
                    else:
                        feature_out[feature_key].append(feature_data[elem])
            else:
                if isinstance(mapping['map'], tuple):
                    feature_out[feature_key] = feature_data[mapping['map'][0]][mapping['map'][1]]
                else:
                    feature_out[feature_key] = feature_data[mapping['map']]
            if 'transform' in mapping:
                feature_out[feature_key] = mapping['transform'](feature_out[feature_key])
        elif feature_key in feature_data:
            feature_out[feature_key] = feature_data[feature_key]
    return feature_out

def filterFeaturesData(features_data, labels, types=None, min_number_points=None, features_point_indices=None):
    if features_point_indices is None:
        features_point_indices = computeFeaturesPointIndices(labels, size=len(features_data))
    
    by_type_condition = lambda x: True
    if types is not None:
        by_type_condition = lambda x: x['type'].lower() in types

    by_npoints_condition = lambda x: True
    if min_number_points is not None:
        by_npoints_condition = lambda x: len(x) >= min_number_points

    labels_map = np.arange(len(features_data), dtype=np.int32)
    new_features_data = [None]*len(features_data)
    new_features_point_indices = [None]*len(features_data)
    for j in range(len(features_data)):
        feature = features_data[j]
        fpi = features_point_indices[j]
        if feature is not None and fpi is not None:
            if by_type_condition(feature) and by_npoints_condition(fpi):
                new_features_data[j] = features_data[j]
                new_features_point_indices[j] = features_point_indices[j]
            else:
                labels_map[j] = -1
        else:
            labels_map[j] = -1

    if labels is not None:
        not_minus_one_mask = labels != -1
        labels[not_minus_one_mask] = labels_map[labels[not_minus_one_mask]]

    return new_features_data, labels, new_features_point_indices

def computeFeaturesPointIndices(labels, size=None):
    if size is None:
        size = np.max(labels) + 1

    features_point_indices = [None]*size
    
    labels_mask = labels != -1
    not_none_features = []
    for i in range(0, len(labels)):
        if labels_mask[i]:
            if features_point_indices[labels[i]] is None:
                not_none_features.append(labels[i])
                features_point_indices[labels[i]] = [i]
            else:
                features_point_indices[labels[i]].append(i)

    for i in not_none_features:
        features_point_indices[i] = np.array(features_point_indices[i], dtype=np.int64)

    return features_point_indices

def computeLabelsFromFace2Primitive(labels, features_data, max_face=None):
    if max_face is None:
        max_face = np.max(labels)
        for feat in features_data:
            max_face = max(0 if len(feat['face_indices']) == 0 else max(feat['face_indices']), max_face)

    face_2_primitive = np.zeros(shape=(max_face+1,), dtype=np.int32) - 1

    for i, feat in enumerate(features_data):
        face_2_primitive[np.asarray(feat['face_indices'], dtype=np.int64)] = i

    labels = face_2_primitive[labels]

    features_point_indices = [[] for i in range(0, len(features_data) + 1)]
    for i in range(0, len(labels)):
        features_point_indices[labels[i]].append(i)
    features_point_indices.pop(-1)

    for i in range(0, len(features_point_indices)):
        features_point_indices[i] = np.array(features_point_indices[i], dtype=np.int64)
    
    return labels, features_point_indices

def downsampleByPointIndices(pcd, indices, labels, leaf_size):
    curr_pcd = pcd.select_by_index(indices)
    curr_pcd, _, id_map = curr_pcd.voxel_down_sample_and_trace(leaf_size, curr_pcd.get_min_bound(), curr_pcd.get_max_bound())
    curr_labels = labels[indices]
    
    #voting scheme: it is needed to recover the face_index information for each point (possible FIXME for the future)
    #each new point is generate by some points from original pcd, so we are using the most voted face_index
    vote_map = [list(curr_labels[np.asarray(ids)]) for ids in id_map]
    down_labels = list([ max(map(lambda val: (votes.count(val), val), set(votes)))[1] for votes in vote_map])

    down_pcd = curr_pcd
    return down_pcd, down_labels

def savePCD(filename, points, colors=None, normals=None, labels=None, binary=True):
    axis = ['x', 'y', 'z']
    fields = [(a, np.float32) for a in axis]
    arr = np.copy(points)
    
    if colors is not None:
        print('Color not implemented yet.')
        pass

    if normals is not None:
        fields += [('normal_{}'.format(a), np.float32) for a in axis]
        arr = np.hstack((arr, normals))
    
    if labels is not None:
       fields += [('label', np.uint32)]
       arr = np.hstack((arr, labels[..., None]))

    arr_s = np.core.records.fromarrays(arr.transpose(), dtype=fields)

    pc = pypcd.PointCloud.from_array(arr_s)

    metadata = pc.get_metadata()
    if binary:
        metadata['data'] = 'binary'
    else:
        metadata['data'] = 'ascii'

    header = pypcd.write_header(metadata)

    with open(filename, 'wb' if binary else 'w') as f:
        if binary:
            f.write(bytes(header.encode('ascii')))
            f.write(arr_s.tobytes())
        else:
            f.write(header)
            f.write('\n'.join([' '.join(map(str, x)) for x in arr_s]))

def computeRotation(axis, theta):
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    
    R = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                    [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                    [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    return R

def projectRays(rays, up, view_dir):
    shp = rays.shape
    
    x_axis = np.array([1, 0, 0])

    rot_axis = np.cross(x_axis, view_dir)
    angle = np.arccos(np.dot(x_axis, view_dir))
    r = computeRotation(rot_axis, angle)
    new_z = r.dot(np.array([0, 0, 1]))

    # Find the projections of the vectors onto the axis
    proj1 = np.dot(new_z, view_dir)
    proj2 = np.dot(up, view_dir)

    # Find the components of the vectors that are perpendicular to the axis
    perp1 = new_z - proj1 * view_dir
    perp2 = up - proj2 * view_dir

    # Find the vector perpendicular to both components
    perp = np.cross(perp1, perp2)

    # Find the magnitude of the perpendicular vector
    mag = np.linalg.norm(perp)

    # Find the scalar product of the two vectors
    scalar_prod = np.dot(new_z, up)

    # Compute the angle between the two vectors with respect to the arbitrary axis
    angle2 = 2*np.pi - np.arctan2(mag, scalar_prod)

    r2 = computeRotation(view_dir, angle2)

    rot = r2.dot(r)

    rays_new = (rot.dot(rays.reshape(-1, 3).T).T).reshape(shp)

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(np.concatenate((rays_new.reshape(-1, 3),
    #                                                         np.array([up*v for v in np.linspace(0, 1, num=1000)]),
    #                                                         np.array([view_dir*v for v in np.linspace(0, 0.5, num=1000)]))))

    # o3d.visualization.draw_geometries([pcd])

    return rays_new

def createRaysLidar(v_fov, h_fov, v_res, h_res, center, eye, up):
    width_px = int(np.ceil(h_fov/h_res))
    height_px = int(np.ceil(v_fov/v_res))

    center_arr = np.asarray(center)
    eye_arr = np.asarray(eye)
    up_arr = np.asarray(up)

    h_axis = up_arr/np.linalg.norm(up_arr)
    ec = center_arr - eye_arr
    ec /= np.linalg.norm(ec)

    h_fov_rad = h_fov*np.pi/180.
    v_fov_rad = v_fov*np.pi/180.

    uls = np.linspace(-h_fov_rad/2, h_fov_rad/2, num=width_px)
    vls = np.linspace(-v_fov_rad/2, v_fov_rad/2, num=height_px)

    vs, us = np.meshgrid(vls, uls)

    us_cos = np.cos(us)[:, :, np.newaxis]
    us_sin = np.sin(us)[:, :, np.newaxis]
    vs_cos = np.cos(vs)[:, :, np.newaxis]
    vs_sin = np.sin(vs)[:, :, np.newaxis]

    rays = np.zeros((width_px, height_px, 6), dtype=np.float32)

    rays[:, :, :3] = eye_arr

    rays_local = np.concatenate((vs_cos*us_cos, vs_cos*us_sin, vs_sin), axis=2)
    norm = np.linalg.norm(rays_local, axis=2)[:, :, np.newaxis]
    rays_local/= np.concatenate((norm, norm, norm), axis=2)

    rays[:, :, 3:] = projectRays(rays_local, h_axis, ec)

    # rots_horz = [Rotation.from_rotvec(u*h_res_rad*h_axis).as_matrix() for u in range(int(-width_px/2), int(width_px/2))]
    # rots_vert_ec = [Rotation.from_rotvec(v*v_res_rad*v_axis).as_matrix().dot(ec) for v in range(int(-height_px/2), int(height_px/2))]

    # rays = np.zeros((width_px, height_px, 6), dtype=np.float32)

    # rays[:, :, :3] = eye_arr

    # for u in range(width_px):
    #     R_horz = rots_horz[u]
    #     rays[u, :, 3:] = np.asarray([R_horz.dot(ec) for ec in rots_vert_ec])
        
    return o3d.core.Tensor(rays)

def ellipsoid_fit(X):
    x1, y1, z1 = X[0]
    x2, y2, z2 = X[1]
    x3, y3, z3 = X[2]

    # Check if the three points are coplanar
    v1 = np.array([x2 - x1, y2 - y1, z2 - z1])
    v2 = np.array([x3 - x1, y3 - y1, z3 - z1])
    cp = np.cross(v1, v2)
    if np.allclose(cp, [0, 0, 0]):
        raise ValueError("The three points are coplanar.")

    denominator_a = y1**2+z1**2
    denominator_b = x1**2+z1**2
    denominator_c = x1**2+y1**2

    if denominator_a == 0 or denominator_b == 0 or denominator_c == 0:
        raise ValueError("The three points are collinear.")


    def ellipsoid_equations(p):
        a, b, c = p
        e1 = (x1**2/a**2 + y1**2/b**2 + z1**2/c**2 - 1)
        e2 = (x2**2/a**2 + y2**2/b**2 + z2**2/c**2 - 1)
        e3 = (x3**2/a**2 + y3**2/b**2 + z3**2/c**2 - 1)
        return np.array([e1, e2, e3])

    a, b, c = least_squares(ellipsoid_equations, np.ones(3)).x

    #print("Semi-axes lengths: a={}, b={}, c={}".format(a, b, c))

    return a, b, c


def createViews(bbox, cell_size=6, distance=2, distance_std=0, min_ground_distance=0.5, spherical=False, view_cell_size=2):
    '''
    A dome is created above the mesh and views are grid sampled in the dome

    Params:
    - cell_size: each area of cell_size X cell_size in the dome will recieve a view point
    - distance: the distance from the center of mesh to the dome surface is half of the greatest lenght of mesh plus the distance param
    - distance_std: standart deviation of the noise in sensor height

    Return:
    - views(NX6):
        - view_points (NX:3)
        - up_direction (NX3:)
    '''

    bbox_vertices = np.asarray(bbox.get_box_points())
    center_ground = bbox.get_center()
    center_ground[2] = 0
    distances_from_center = np.asarray([np.linalg.norm(vertex-center_ground) for vertex in bbox_vertices])
    farthest_vertex = bbox_vertices[np.argmax(distances_from_center)]
    farthest_vertex = farthest_vertex + distance*(farthest_vertex/np.linalg.norm(farthest_vertex))    

    if not spherical:
        p1 = farthest_vertex
        p2 = farthest_vertex.copy()
        p2[0] = -p2[0]
        p3 = farthest_vertex.copy()
        p3[1] = -p3[1]

        X =  np.concatenate((p1[np.newaxis, :], p2[np.newaxis, :], p3[np.newaxis, :]), axis=0)
        radii = np.asarray(list(ellipsoid_fit(X)))
    else:
        radii = np.zeros(3) + np.linalg.norm(farthest_vertex)

    LH = np.pi*radii[2]/2

    num_cells_h = max(2, int(np.ceil(LH/cell_size)))
    ground_angle = -np.pi/2 + np.arctan2(min_ground_distance, radii[2])
    vs = np.linspace(ground_angle, 0, num=num_cells_h)

    dome_points = []
    for i in range(len(vs)):
        a, b, c = np.sin(vs[i])*radii[0], np.sin(vs[i])*radii[1], radii[2]
        a_abs, b_abs = abs(a), abs(b)
        LG = np.pi*(1.5*(a_abs + b_abs) - np.sqrt(a_abs*b_abs)) + np.finfo(np.float64).eps
        num_cells_g = int(np.ceil(LG/cell_size))

        if i < (len(vs) - 1):
            num_cells_g = max(4, num_cells_g)

        us = np.linspace(0, 2*np.pi - 2*np.pi/num_cells_g, num=num_cells_g)

        xs = a*np.cos(us)[:, np.newaxis]
        ys = b*np.sin(us)[:, np.newaxis]
        zs = (np.ones(num_cells_g)*c*np.cos(vs[i]))[:, np.newaxis]

        dome_points += np.concatenate((xs, ys, zs), axis=1).tolist()

    dome_points = np.asarray(dome_points)
        
    bb_floor_center = bbox.get_center()
    bb_floor_center[2] = bbox.get_min_bound()[2]

    dome_normals = bb_floor_center - dome_points
    dome_normals /= np.linalg.norm(dome_normals, axis=0)
        
    #print('Dome Points:', dome_points)

    #print('Dome Normals:', dome_normals)

    views = np.zeros((len(dome_points), 6), dtype=np.float64)
    #adding gaussian noise to the height of the view
    views[:, :3] = dome_points + dome_normals*np.random.normal(0.0, distance_std, len(views))[:, np.newaxis]

    #print('Views Shape:', views.shape)

    #print('Views Without Up:', views)

    #computing up points to the views (I have never done this, it is needed to test)
    size = bbox.get_max_bound() - bbox.get_min_bound()
    bb_diagonal_dir = size/np.linalg.norm(size)
    ortg_vectors = -(dome_normals*(np.sum(bb_diagonal_dir*dome_normals, axis=1)[:, np.newaxis])) + bb_diagonal_dir
    ortg_vectors /= np.linalg.norm(ortg_vectors, axis=0)
    views[:, 3:] = ortg_vectors#np.cross(dome_points, ortg_vectors)
    #views[:, 3:] /= np.linalg.norm(views[:, 3:])


    #generating dome lines for visualization purpose
    radii = np.abs(radii)
    LG = np.pi*(1.5*(radii[0] + radii[1]) - np.sqrt(radii[0]*radii[1])) + np.finfo(np.float64).eps
    num_view_cells_h = max(2, int(np.ceil(LH/view_cell_size)))
    num_view_cells_g = max(4, int(np.ceil(LG/view_cell_size)))

    view_us_ls = np.linspace(0, 2*np.pi, num_view_cells_g)
    view_vs_ls = np.linspace(-np.pi/2, 0, num_view_cells_h)

    view_us, view_vs = np.meshgrid(view_us_ls, view_vs_ls)

    view_vertices = np.concatenate(((radii[0]*np.cos(view_us)*np.sin(view_vs))[:, :, np.newaxis],
                                    (radii[1]*np.sin(view_us)*np.sin(view_vs))[:, :, np.newaxis],
                                    (radii[2]*np.cos(view_vs))[:, :, np.newaxis]), axis=2)
    
    view_lines = []
    for i in range(view_vertices.shape[0] - 1):
        for j in range(view_vertices.shape[1]):
            i_nxt = i + 1
            j_nxt = j + 1 if (j + 1) <= view_vertices.shape[1] else 0
            fn = lambda x, y: x*view_vertices.shape[1] + y
            view_lines.append([fn(i,j), fn(i_nxt,j)])
            view_lines.append([fn(i,j), fn(i,j_nxt)])
            view_lines.append([fn(i_nxt,j_nxt), fn(i_nxt,j)])
            view_lines.append([fn(i_nxt,j_nxt), fn(i,j_nxt)])

    dome_colors = [[131/255., 148/255., 161/255.] for i in range(len(view_lines))]
    dome_lines = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(view_vertices.reshape((-1, 3))),
        lines=o3d.utility.Vector2iVector(view_lines),
    )
    dome_lines.colors = o3d.utility.Vector3dVector(dome_colors)

    return views, dome_lines

def IDS2RGB(ids):
    ids_arr = np.asarray(ids, dtype=np.uint32)
    r = (0xFF & ids_arr).view(np.float32)[:, np.newaxis]
    g = (0xFF & (ids_arr>>8)).view(np.float32)[:, np.newaxis]
    b = (0xFF & (ids_arr>>16)).view(np.float32)[:, np.newaxis]
    return np.concatenate((r, g, b), axis=1).astype(np.float64)

def RGB2IDS(rgb):
    rgb_arr = np.asarray(rgb, dtype=np.float64).astype(np.float32)
    ids = (rgb_arr[:, 2].view(np.uint32)<<16) | (rgb_arr[:, 1].view(np.uint32)<<8) | rgb_arr[:, 0].view(np.uint32)
    return ids

def pairWiseRegistration(source, target, source_labels, target_labels, distance_threshold=0.005):
    source_tree = o3d.geometry.KDTreeFlann(source)
    _, indices, distances = zip(*[source_tree.search_knn_vector_3d(point, 1) for point in target.points])
    indices = np.asarray(indices)[:, 0]
    distances = np.sqrt(np.asarray(distances)[:, 0])

    #source_colors = np.asarray(source.colors)[indices]
    source_ids = source_labels[indices]
    #source_ids = RGB2IDS(source_colors)
    #target_colors = np.asarray(target.colors)
    #target_ids = RGB2IDS(target_colors)
    target_ids = target_labels.copy()

    valid_indices = np.arange(len(distances))[np.logical_or(distances > distance_threshold, source_ids != target_ids)]

    target_outlier_pcd = target.select_by_index(valid_indices)
    target_outlier_labels = target_labels[valid_indices]

    source += target_outlier_pcd
    source_labels = np.concatenate((source_labels, target_outlier_labels))
    

    return source, source_labels

LIDAR_KEYS =['vertical_fov', 'horizontal_fov', 'vertical_resolution', 'horizontal_resolution']

def rayCastingPointCloudGeneration(mesh, lidar_data={'vertical_fov':180, 'horizontal_fov':180,
                                                     'vertical_resolution':0.09, 'horizontal_resolution':0.09},
                                   dome_cell_size=1, distance_std=0., distance=0.5, verbose=True, view_pcd=False):
    

    assert np.array([key in lidar_data.keys() for key in LIDAR_KEYS]).all(), 'Missing keys in lidar_data dictionary.'

    vertical_fov = lidar_data['vertical_fov']
    horizontal_fov = lidar_data['horizontal_fov']
    vertical_resolution = lidar_data['vertical_resolution']
    horizontal_resolution = lidar_data['horizontal_resolution']
    
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    bbox = mesh.get_axis_aligned_bounding_box()

    funif(print, verbose)('Generating Views...')
    views, dome_lines = createViews(bbox, distance=distance, cell_size=dome_cell_size, distance_std=distance_std)
    funif(print, verbose)('Done.\n')

    if view_pcd:
        bbox_size = bbox.get_max_bound() - bbox.get_min_bound()
        view_lookat = np.array([0, -0.25*bbox_size[1], 1.3*distance + bbox_size[2]])
        view_front = view_lookat #-  bbox.get_center()
        view_params = {'lookat': view_lookat, 
                       'up': np.array([0, 0, 1]), 
                       'front': view_front/np.linalg.norm(view_front), 'zoom': 0.3}
        
        o3d.visualization.draw_geometries([mesh, dome_lines], mesh_show_wireframe=True, **view_params)
        drone_mesh = o3d.io.read_triangle_mesh('./resources/drone.stl')
        vertices_arr = np.asarray(drone_mesh.vertices)
        drone_mesh.vertices = o3d.utility.Vector3dVector((vertices_arr - np.mean(vertices_arr, axis=0))*2)
        colors = np.asarray(get_evenly_distributed_colors(len(views)))/255.
        drones = []
        for i, view in enumerate(views):
            drone_curr = deepcopy(drone_mesh)
            vertices_arr_curr = np.asarray(drone_curr.vertices)
            drone_curr.vertices = o3d.utility.Vector3dVector(vertices_arr_curr + view[:3])
            drone_curr.vertex_colors = o3d.utility.Vector3dVector(np.zeros((len(vertices_arr_curr), 3)) + colors[i])
            drones.append(drone_curr)
        o3d.visualization.draw_geometries([mesh, dome_lines] + drones, **view_params)

    funif(print, verbose)('Generating Rays...')
    multi_view_rays = [createRaysLidar(vertical_fov, horizontal_fov, vertical_resolution, horizontal_resolution,
                                       bbox.get_center(), view[:3], view[3:]) for view in funif(tqdm, verbose)(views)]
    funif(print, verbose)('Done.\n')
        
    funif(print, verbose)('Casting Rays and Registering...')
    registered_pcd = None
    registered_labels_mesh = None
    for i, rays in enumerate(funif(tqdm, verbose)(multi_view_rays)):
        ans = scene.cast_rays(rays)

        hit = ans['t_hit'].isfinite()

        points = rays[hit][:,:3] + rays[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1))
        normals = ans['primitive_normals'][hit].numpy()

        normals = normals/(np.linalg.norm(normals, axis=1)[:, np.newaxis])

        labels_mesh = ans['primitive_ids'][hit].numpy()

        if view_pcd:
            color = colors[i]
            pcd_view = o3d.geometry.PointCloud()
            pcd_view.points = o3d.utility.Vector3dVector(points.numpy())
            pcd_view.normals = o3d.utility.Vector3dVector(normals)
            pcd_view.colors = o3d.utility.Vector3dVector(np.zeros((len(normals),  3)) + color)
            drones_pcd = o3d.geometry.PointCloud()
            indices = np.concatenate((np.arange(0, i), np.arange(i + 1, len(views)))) 
            drones_pcd.points = o3d.utility.Vector3dVector(views[indices, :3])
            drones_pcd.colors = o3d.utility.Vector3dVector(colors[indices])
            o3d.visualization.draw_geometries([pcd_view, mesh, dome_lines, drones_pcd, drones[i]], **view_params)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.numpy())
        pcd.normals = o3d.utility.Vector3dVector(normals)
        #pcd.colors = o3d.utility.Vector3dVector(rgb)

        if i == 0:
            registered_pcd = pcd
            registered_labels_mesh = labels_mesh
        else:
            registered_pcd, registered_labels_mesh = pairWiseRegistration(registered_pcd, pcd, registered_labels_mesh, labels_mesh)
            # result_icp = o3d.pipelines.registration.registration_colored_icp(pcd, registered_pcd,
            #             0.005, estimation_method=o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            #             criteria=o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
            #                                                                        relative_rmse=1e-6,
            #                                                                        max_iteration=30))
        
            # corr = np.asarray(result_icp.correspondence_set)
            # pcd_inliers = registered_pcd.select_by_index(corr[:, 1])
            # pdc_1_outliers = registered_pcd.select_by_index(corr[:, 1], invert=True)
            # pcd_2_outliers = pcd.select_by_index(corr[:, 0], invert=True)
            # registered_pcd = pcd_inliers + pdc_1_outliers + pcd_2_outliers
        
    if view_pcd:
        o3d.visualization.draw_geometries([registered_pcd, mesh], **view_params)

    #registered_labels_mesh = RGB2IDS(registered_pcd.colors)
    funif(print, verbose)('Done.\n')

    return registered_pcd, registered_labels_mesh