import numpy as np
import pickle
import json
import yaml
from numba import njit

from os.path import exists
from os import system

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


'''POINT CLOUDS'''

def generatePCD(pc_filename, mps_ns, mesh_filename=None): 
    if not exists(pc_filename):
        if mesh_filename is None:
            return []
        system(f'mesh_point_sampling {mesh_filename} {pc_filename} --n_samples {mps_ns} --write_normals --no_vis_result > /dev/null')

    return True


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

def filterFeaturesData(features_data, surface_types, labels=None):
    labels_map = np.zeros(len(features_data))
    i = 0
    j = 0
    while i < len(features_data):
        feature = features_data[i]
        if feature['type'].lower() not in surface_types:
            features_data.pop(i)
            labels_map[j] = -1
        else:
            labels_map[j] = i
            i+=1
        j+=1

    if labels is not None:
        for i in range(len(labels)):
            labels[i] = labels_map[labels[i]]
        return features_data, labels

    return features_data

def computeFeaturesPointIndices(labels):
    size = np.max(labels)
    features_point_indices = [[] for i in range(0, len(size) + 1)]
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