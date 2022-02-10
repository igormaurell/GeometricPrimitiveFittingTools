from tqdm import tqdm

from os import listdir, mkdir, system
from os.path import isfile, join, exists

import pickle

import h5py

import numpy as np

from pypcd import pypcd

import gc

from tools import loadFeatures, filterFeaturesData

from normalization import centralize, add_noise

def generateFace2PrimitiveMap(features_data, max_face=0):
    for feat in features_data:
        max_face = max(0 if len(feat['face_indices']) == 0 else max(feat['face_indices']), max_face)
    face_2_primitive = np.zeros(shape=(max_face+1,), dtype=np.int32) - 1
    face_primitive_count = np.zeros(shape=(max_face+1,), dtype=np.int32)
    for i, feat in enumerate(features_data):
        for face in feat['face_indices']:
            face_2_primitive[face] = i
            face_primitive_count[face] += 1
    if len(np.unique(face_primitive_count)) > 2:
        print('There is faces that lies to more than one primitive.')
    return face_2_primitive

LSSPFN_FEATURES = {
    'plane': ['location', 'z_axis', 'normalized'],
    'cylinder': ['location', 'z_axis', 'radius', 'normalized'],
    'cone': ['location', 'z_axis', 'radius', 'angle', 'apex', 'normalized'],
    'sphere': ['location', 'radius', 'normalized']
}

def filterFeature2LSSPFN(feature, name):
    tp = feature['type'].lower()
    if tp in LSSPFN_FEATURES.keys():
        feature_out = {}
        feature_out['name'] = name
        feature_out['type'] = tp
        for field in LSSPFN_FEATURES[tp]:
            feature_out[field] = feature[field]
        return feature_out

    feature['name'] = name
    feature['type'] = tp
    return feature

def normalize2LSSPFN(point_cloud, features=[], noise_limit = 10):
    point_cloud, features = centralize(point_cloud, features)
    if noise_limit != 0:
       point_cloud = add_noise(point_cloud, limit=noise_limit)
    for i in range(0, len(features)):
        features[i]['normalized'] = True
    return point_cloud, features

def generatePCD2LSSPFN(pc_filename, mps_ns, mesh_filename=None): 
    if not exists(pc_filename):
        if mesh_filename is None:
            return []
        system(f'mesh_point_sampling {mesh_filename} {pc_filename} --n_samples {mps_ns} --write_normals --no_vis_result > /dev/null')

    return True

def generateH52LSSPFN(pc_filename, features_data, h5_filename, noise_limit):
    h5_file = h5py.File(h5_filename, 'w')
        
    pc = pypcd.PointCloud.from_path(pc_filename).pc_data

    gt_points = np.ndarray(shape=(pc['x'].shape[0], 3), dtype=np.float64)
    gt_points[:, 0] = pc['x']
    gt_points[:, 1] = pc['y']
    gt_points[:, 2] = pc['z']

    gt_normals = np.ndarray(shape=(pc['x'].shape[0], 3), dtype=np.float64)
    gt_normals[:, 0] = pc['normal_x']
    gt_normals[:, 1] = pc['normal_y']
    gt_normals[:, 2] = pc['normal_z']

    point_cloud = np.concatenate((gt_points, gt_normals), axis=1)

    del gt_points
    del gt_normals
    gc.collect()

    gt_point_cloud, features_data = normalize2LSSPFN(point_cloud.copy(), features=features_data, noise_limit=0)

    h5_file.create_dataset('gt_points', data=gt_point_cloud[:, :3])
    h5_file.create_dataset('gt_normals', data=gt_point_cloud[:, 3:])

    del gt_point_cloud
    gc.collect()
    
    gt_labels = pc['label']

    face_2_primitive = generateFace2PrimitiveMap(features_data, max_face=np.max(gt_labels))

    features_points = [[] for i in range(0, len(features_data))]
    for i in range(0, len(gt_labels)):
        index = face_2_primitive[gt_labels[i]]
        if index != -1:
            features_points[index].append(i)
        gt_labels[i] = index
    
    h5_file.create_dataset('gt_labels', data=gt_labels)

    del pc
    del gt_labels
    gc.collect()

    noisy_point_cloud, _ = normalize2LSSPFN(point_cloud, noise_limit=noise_limit) 

    h5_file.create_dataset('noisy_points', data=noisy_point_cloud[:, :3])

    del noisy_point_cloud
    gc.collect()

    point_position = h5_filename.rfind('.')
    point_position = point_position if point_position >= 0 else len(point_position)
    bar_position = h5_filename.rfind('/')
    bar_position = bar_position if bar_position >= 0 else 0

    name = h5_filename[bar_position+1:point_position]

    for i, feature in enumerate(features_data):
        soup_name = f'{name}_soup_{i}'
        grp = h5_file.create_group(soup_name)
        gt_indices = np.array(features_points[i])
        grp.create_dataset('gt_indices', data=gt_indices)
        feature = filterFeature2LSSPFN(feature, soup_name)
        grp.attrs['meta'] = np.void(pickle.dumps(feature))

    h5_file.close()

def generateLSSPFN(features_folder_name, mesh_folder_name, pc_folder_name, h5_folder_name, mps_ns, noise_limit, surface_types):
    if exists(features_folder_name):
        features_files = sorted([f for f in listdir(features_folder_name) if isfile(join(features_folder_name, f))])
        print(f'\nGenerating dataset for {len(features_files)} features files...\n')
    else:
        print('\nThere is no features folder.\n')
        return False
    
    if not exists(h5_folder_name):
        mkdir(h5_folder_name)

    for features_filename in tqdm(features_files):
        print(features_filename )
        point_position = features_filename.rfind('.')
        filename = features_filename[:point_position]

        pc_filename = join(pc_folder_name, filename) + '.pcd'
        mesh_filename = join(mesh_folder_name, filename) + '.obj'
              
        if exists(pc_filename):
            generatePCD2LSSPFN(pc_filename, mps_ns)

        elif exists(mesh_filename):
            if not exists(pc_folder_name):
                mkdir(pc_folder_name)

            generatePCD2LSSPFN(pc_filename, mps_ns, mesh_filename=mesh_filename)

        else:
            print(f'\nFeature {filename} has no PCD or OBJ to use.')
            continue

        feature_ext =  features_filename[(point_position + 1):]
        features_data = loadFeatures(join(features_folder_name, filename), feature_ext)
        filterFeaturesData(features_data, [], surface_types)

        h5_filename = join(h5_folder_name, f'{filename}.h5')
        
        generateH52LSSPFN(pc_filename, features_data['surfaces'], h5_filename, noise_limit)
    
    print()
    return True