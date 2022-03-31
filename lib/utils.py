import numpy as np
import pickle
import json
import yaml
from numba import njit
from os.path import exists
from os import system
from math import acos

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
    c = abs(np.dot(n1.T, n2))
    c = c if c <= 1. else 1.
    c = c if c >= -1. else -1. 
    return acos(c)


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

def filterFeature(feature_data, features_by_type, features_translation):
    assert 'type' in feature_data.keys()
    tp = feature_data['type'].lower()
    assert tp in features_by_type.keys()
    feature_out = {}
    for key in features_by_type[tp]:
        feature_out[key if not key in features_translation.keys() else features_translation[key]] = feature_data[key]
    return feature_out

def filterFeaturesData(features_data, types=None, min_number_points=None, labels=None, features_point_indices=None):
    by_type_condition = lambda x: True
    if types is not None:
        by_type_condition = lambda x: x['type'].lower() in types

    by_npoints_condition = lambda x: True
    if min_number_points is not None and features_point_indices is not None:
        by_npoints_condition = lambda x: len(x) >= min_number_points

    labels_map = np.zeros(len(features_data))
    i = 0
    j = 0
    while i < len(features_data):
        feature = features_data[i]
        fpi = features_point_indices[i]
        if by_type_condition(feature) and by_npoints_condition(fpi):
            labels_map[j] = i
            i+=1
        else:
            features_data.pop(i)
            if features_point_indices is not None:
                features_point_indices.pop(i)
            labels_map[j] = -1
        j+=1

    if labels is not None:
        for i in range(len(labels)):
            if labels[i] != -1:
                labels[i] = labels_map[labels[i]]

    return features_data, labels, features_point_indices

def computeFeaturesPointIndices(labels, size=None):
    if size is None:
        size = np.max(labels)
    features_point_indices = [[] for i in range(0, size + 2)]
    for i in range(0, len(labels)):
        features_point_indices[labels[i]].append(i)
    features_point_indices.pop(-1)

    for i in range(0, len(features_point_indices)):
        features_point_indices[i] = np.array(features_point_indices[i], dtype=np.int64)

    return features_point_indices

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

