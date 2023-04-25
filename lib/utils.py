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
from scipy.spatial.transform import Rotation

EPS = np.finfo(np.float64).eps

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

'''COLOR'''
def computeRGB(value):
    r = value%256
    value = value//256
    g = value%256
    b = value//256
    return (r, g, b)

def getAllColorsArray():
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

def filterFeaturesData(features_data, types=None, min_number_points=None, labels=None, features_point_indices=None):
    by_type_condition = lambda x: True
    if types is not None:
        by_type_condition = lambda x: x['type'].lower() in types
    by_npoints_condition = lambda x: True
    if min_number_points is not None and features_point_indices is not None:
        by_npoints_condition = lambda x: len(x) >= min_number_points

    labels_map = np.zeros(len(features_data))
    new_features_data = []
    new_features_point_indices = []
    i = 0
    for j in range(len(features_data)):
        feature = features_data[j]
        fpi = features_point_indices[j]
        labels_map[j] = -1
        if by_type_condition(feature) and by_npoints_condition(fpi):
            labels_map[j] = i
            new_features_data.append(features_data[j])
            new_features_point_indices.append(features_point_indices[j])
            i+= 1            

    if labels is not None:
        for i in range(len(labels)):
            if labels[i] != -1:
                labels[i] = labels_map[labels[i]]

    return new_features_data, labels, new_features_point_indices

def computeFeaturesPointIndices(labels, size=None):
    if size is None:
        size = np.max(labels) + 1
    features_point_indices = [[] for i in range(0, size + 1)]
    for i in range(0, len(labels)):
        features_point_indices[labels[i]].append(i)
    features_point_indices.pop(-1)

    for i in range(0, len(features_point_indices)):
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

# too slow, python problem???
def samplePointsUniformlyAndTrack(mesh, number_of_points, use_triangle_normal=False):
    surface_area = mesh.get_surface_area()
    if surface_area <= 0:
        print("ERROR: Invalid surface area {}, it must be > 0.".format(surface_area))

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)
    triangle_normals = np.zeros(triangles.shape, dtype=np.float64)
    triangle_areas = np.zeros(len(mesh.triangles), dtype=np.float64)
    from tqdm import tqdm

    for i, t in tqdm(enumerate(triangles)):
        A = vertices[t[0]]
        B = vertices[t[1]]
        C = vertices[t[2]]

        #normal (counter-clockwise as default order)
        v1 = A - C
        v2 = B - C
        normal = np.cross(v1, v2)
        triangle_normals[i] = normal

        #area
        area = np.linalg.norm(np.cross(v1, v2))/2
        triangle_areas[i] = area        

    triangle_areas[0] /= surface_area
    for i in range(1, len(triangles)):
        triangle_areas[i] = triangle_areas[i] / (surface_area + triangle_areas[i - 1])

    has_vert_normal = mesh.has_vertex_normals()
    has_vert_color = mesh.has_vertex_colors()

    points = np.array((number_of_points, 3), dtype=np.float64)
    normals = np.array((number_of_points, 3), dtype=np.float64)
    colors = np.array((number_of_points, 3), dtype=np.float64)
    points_label = np.array(number_of_points, dtype=np.int32)

    vertex_normals = mesh.vertex_normals
    vertex_colors = mesh.vertex_colors

    last_n = 0
    for tidx, t in tqdm(enumerate(triangles)):
        n = int(np.around(triangle_areas[tidx] * number_of_points))
        randoms1 = np.random.uniform(size=n)
        randoms2 = np.random.uniform(size=n)

        a = 1 - np.sqrt(randoms1)
        b = np.sqrt(randoms1) * (1 - randoms2)
        c = np.sqrt(randoms1) * randoms2

        V1, V2, V3 = vertices[t]

        points[last_n:n, :] =  a*V1 + b*V2 + c*V3

        if has_vert_normal and not use_triangle_normal:
            n1, n2, n3 = vertex_normals[t]
            normals[last_n:n, :] = a*n1 + b*n2 + c*n2
        elif use_triangle_normal:
            normals[last_n:n, :] = triangle_normals[tidx]
        
        if has_vert_color:
            c1, c2, c3 = vertex_colors[t]
            normals[last_n:n, :] = a*c1 + b*c2 + c*c3
        
        points_label[last_n:n] = tidx

        last_n = n

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if has_vert_normal or use_triangle_normal:
        normals = normals/(np.linalg.norm(normals, axis=0) + EPS)
        pcd.normals = o3d.utility.Vector3dVector(normals)
    
    if has_vert_color:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def createRaysLidar(v_fov, h_fov, v_res, h_res, center, eye, up):
    width_px = int(np.ceil(h_fov/h_res))
    height_px = int(np.ceil(v_fov/v_res))

    center_arr = np.asarray(center)
    eye_arr = np.asarray(eye)
    up_arr = np.asarray(up)

    h_axis = up_arr/np.linalg.norm(up_arr)
    ec = center_arr - eye_arr
    ec /= np.linalg.norm(ec)
    v_axis = np.cross(ec, h_axis)

    h_res_rad = h_res*np.pi/180.
    v_res_rad = v_res*np.pi/180.

    lidar_data = np.zeros((width_px, height_px, 6), dtype=np.float32)

    for u in range(width_px):
        u_norm = u - width_px/2
        h_angle = u_norm*h_res_rad
        R = Rotation.from_rotvec(h_angle*h_axis)
        for v in range(height_px):
            v_norm = v - height_px/2
            v_angle = v_norm*v_res_rad
            R_curr = Rotation.from_rotvec(v_angle*v_axis)
            R_curr = R.as_matrix().dot(R_curr.as_matrix())
            direction = R_curr.dot(ec)
            lidar_data[u, v, :3] = eye_arr
            lidar_data[u, v, 3:] = direction
    
    return o3d.core.Tensor(lidar_data)

def rayCastingPointCloudGeneration(mesh, distance=10, vertical_fov=100, horizontal_fov=180, vertical_resolution=0.1, horizontal_resolution=0.1):
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    bounding_box = mesh.get_axis_aligned_bounding_box()

    bb_min, bb_max = bounding_box.get_min_bound(), bounding_box.get_max_bound()
    bb_half_size = bounding_box.get_half_extent()
    bb_center = bounding_box.get_center()
    import time
    a = time.time()
    # #for laterals
    multi_view_rays = []
    bb_center_lat = [np.array([(bb_half_size[0] + distance)*i, (bb_half_size[1] + distance)*j, bb_half_size[2]]) + bb_min for i, j in ((0,0), (0,1), (1,0), (1,1))]
    for point in bb_center_lat:
        rays = createRaysLidar(
            vertical_fov,
            horizontal_fov,
            vertical_resolution,
            horizontal_resolution,
            bb_center,
            point,
            [0, 0, 1]
        )
        multi_view_rays.append(rays)

    bb_center_top  = [np.array([bb_half_size[0], bb_half_size[1], 2*bb_half_size[2] + distance]) + bb_min]
    for point in bb_center_top:
        rays = createRaysLidar(
            vertical_fov,
            horizontal_fov,
            vertical_resolution,
            horizontal_resolution,
            bb_center,
            point,
            [0 if bb_half_size[0] > bb_half_size[1] else 1, 1 if bb_half_size[0] > bb_half_size[1] else 0 , 0],
        )
        multi_view_rays.append(rays)

    print(time.time() - a)
    
    registered_pcd = o3d.geometry.PointCloud()
    registered_labels_mesh = []
    for i, rays in enumerate(multi_view_rays):
        print(registered_pcd)

        ans = scene.cast_rays(rays)

        hit = ans['t_hit'].isfinite()

        points = rays[hit][:,:3] + rays[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1))
        print(len(points))
        normals = ans['primitive_normals'][hit].numpy()

        normals = normals/np.linalg.norm(normals)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.numpy())
        pcd.normals = o3d.utility.Vector3dVector(normals)

        labels_mesh = ans['primitive_ids'][hit].numpy()
        
        if i > 0:
            registration_result = o3d.pipelines.registration.registration_icp(registered_pcd, pcd, 0.03)
            corr = np.asarray(registration_result.correspondence_set)
            pcd_inliers = registered_pcd.select_by_index(corr[:, 0])
            pdc_1_outliers = registered_pcd.select_by_index(corr[:, 0], invert=True)
            pcd_2_outliers = pcd.select_by_index(corr[:, 1], invert=True)
            registered_pcd = pcd_inliers + pdc_1_outliers + pcd_2_outliers
            
            lm_1_inliers = registered_labels_mesh[corr[:, 0]]
            lm_2_inliers = labels_mesh[corr[:, 1]]
            matches = lm_1_inliers == lm_2_inliers
            count_matches = np.count_nonzero(~matches)
            if count_matches > 0:
                print('Error in labels mesh registration, {} points have differente triangle ids.'.format(count_matches))
            corr_1 = np.in1d(range(registered_labels_mesh.shape[0]), corr[:, 0])
            lm_1_outliers = registered_labels_mesh[~corr_1]
            corr_2 = np.in1d(range(labels_mesh.shape[0]), corr[:, 1])
            lm_2_outliers = labels_mesh[~corr_2]
            registered_labels_mesh = np.concatenate((lm_1_inliers, lm_1_outliers, lm_2_outliers))

        else:
            registered_pcd = pcd
            registered_labels_mesh = labels_mesh

    return registered_pcd, np.asarray(registered_labels_mesh)