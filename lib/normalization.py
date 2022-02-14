import numpy as np

FEATURE_TYPE = {
    'location'  : 'point',
    'direction' : 'vector',
    'z_axis'    : 'vector',
    'radius'    : 'value',
    'x_axis'    : 'vector',
    'y_axis'    : 'vector',
    'focus1'    : 'point',
    'focus2'    : 'point',
    'x_radius'  : 'value',
    'y_radius'  : 'value',
    'apex'      : 'point',
    'max_radius': 'value',
    'min_radius': 'value'
}

EPS = np.finfo(np.float32).eps

def rotation_matrix_a_to_b(A, B):
    cos = np.dot(A, B)
    sin = np.linalg.norm(np.cross(B, A))
    u = A
    v = B - np.dot(A, B) * A
    v = v / (np.linalg.norm(v) + EPS)
    w = np.cross(B, A)
    w = w / (np.linalg.norm(w) + EPS)
    F = np.stack([u, v, w], 1)
    G = np.array([[cos, -sin, 0],
                [sin, cos, 0],
                [0, 0, 1]])
    # B = R @ A
    try:
        R = F @ G @ np.linalg.inv(F)
    except:
        R = np.eye(3, dtype=np.float32)
    return R

def pca_numpy(array):
    S, U = np.linalg.eig(array.T @ array)
    return S, U

def addNoise(points, normals, limit=0.01):
    noise = normals * np.random.uniform(-limit, limit, (points.shape[0],1))
    points = points + noise.astype(np.float32)
    #not adding noise on normals yet
    return points

def rotateFeatures(features, transform):
    for i in range(0, len(features)):
        for key in features[i].keys():
            if key in FEATURE_TYPE.keys():
                if FEATURE_TYPE[key] == 'point' or FEATURE_TYPE[key] == 'vector':
                    features[i][key] = list((transform @ np.array(features[i][key]).T).T)
    return features

def alignCanonical(points, normals=None, features=[]):
    S, U = pca_numpy(points)
    smallest_ev = U[:, np.argmin(S)]
    R = rotation_matrix_a_to_b(smallest_ev, np.array([1, 0, 0]))
    # rotate input points such that the minor principal
    # axis aligns with x axis.
    points = (R @ points.T).T
    if normals is not None:
        normals = (R @ normals.T).T
    features = rotateFeatures(features, R)
    return points, normals, features, R

def translateFeatures(features, transform):
    for i in range(0, len(features)):
        for key in features[i].keys():
            if key in FEATURE_TYPE.keys():
                if FEATURE_TYPE[key] == 'point':
                    features[i][key] = list(np.array(features[i][key]) + transform)
    return features

def centralize(points, features=[]):
    mean = np.mean(points, 0)
    centralized_points = points - mean
    features = translateFeatures(features, -mean)
    return centralized_points, features, -mean

def reescaleFeatures(features, factor):
    for i in range(0, len(features)):
        for key in features[i].keys():
            if key in FEATURE_TYPE.keys():
                if FEATURE_TYPE[key] == 'point':
                    features[i][key] = list(np.array(features[i][key])*factor)
                if FEATURE_TYPE[key] == 'value':
                    features[i][key]*= factor
    return features

def rescale(points, features=[], factor=1000):
    f = factor + EPS
    scaled_points = points*f
    features = reescaleFeatures(features, f)
    return scaled_points, features, f

def cubeRescale(points, features=[], factor=1):
    std = np.max(points, 0) - np.min(points, 0)
    f = factor/(np.max(std) + EPS)
    scaled_points = points*f
    features = reescaleFeatures(features, f)
    return scaled_points, features, f

def normalize(points, parameters,  normals=None, features=[]):
    transforms = {
        'translation': np.zeros(3),
        'rotation': np.eye(3),
        'scale': 0.,
        'sequence': ['translation', 'rotation', 'scale']
    }

    if 'rescale' in parameters.keys():
        points, features, transforms['scale'] = rescale(points, features, parameters['rescale'])
    if 'centralize' in parameters.keys() and parameters['centralize'] == True:
        points, features, transforms['translation'] = centralize(points, features)
    if 'align' in parameters.keys() and parameters['align'] == True:
        points, normals, features, transforms['rotation'] = alignCanonical(points, normals, features)
    if 'add_noise' in parameters.keys() and parameters['add_noise'] != 0.:
        assert normals is not None
        points = addNoise(points, normals, parameters['add_noise'])
    if 'cube_rescale' in parameters.keys() and parameters['cube_rescale'] > 0:
        points, features, scale = cubeRescale(points, features, parameters['cube_rescale'])
        transforms['scale'] *= scale

    return points, normals, features, transforms