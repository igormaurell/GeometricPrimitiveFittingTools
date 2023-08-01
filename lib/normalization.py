import numpy as np

#TODO: do the normalization again using asGeometryOCCWrapper (smaller code)


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

def rotateUtil(points, transform, normals=None, features=[]):
    points = (transform @ points.T).T
    if normals is not None:
        normals = (transform @ normals.T).T
    for i in range(0, len(features)):
        if features[i] is not None:
            for key in features[i].keys():
                if key in FEATURE_TYPE.keys():
                    if FEATURE_TYPE[key] == 'point' or FEATURE_TYPE[key] == 'vector':
                        features[i][key] = list((transform @ np.array(features[i][key]).T).T)
    return points, normals, features

def alignCanonical(points, normals=None, features=[]):
    S, U = pca_numpy(points)
    smallest_ev = U[:, np.argmin(S)]
    R = rotation_matrix_a_to_b(smallest_ev, np.array([1, 0, 0]))
    points, normals, features = rotateUtil(points, R, normals=normals, features=features)
    return points, normals, features, R

def translateUtil(points, transform, normals=None, features=[]):
    points = points + transform
    for i in range(0, len(features)):
        if features[i] is not None:
            for key in features[i].keys():
                if key in FEATURE_TYPE.keys():
                    if FEATURE_TYPE[key] == 'point':
                        features[i][key] = list(np.array(features[i][key]) + transform)
    return points, normals, features

def centralize(points, features=[]):
    mean = np.mean(points, 0)
    points, _, features = translateUtil(points, -mean, features=features)
    return points, features, -mean

def reescaleUtil(points, factor, normals=None, features=[]):
    points = points*factor
    for i in range(0, len(features)):
        if features[i] is not None:
            for key in features[i].keys():
                if key in FEATURE_TYPE.keys():
                    if FEATURE_TYPE[key] == 'point':
                        features[i][key] = list(np.array(features[i][key])*factor)
                    if FEATURE_TYPE[key] == 'value':
                        features[i][key]*= factor
    return points, normals, features

def rescale(points, features=[], factor=1000):
    f = factor + EPS
    points, _, features = reescaleUtil(points, f, features=features)
    return points, features, f

def cubeRescale(points, features=[], factor=1):
    std = np.max(points, 0) - np.min(points, 0)
    f = factor/(np.max(std) + EPS)
    points, _, features = reescaleUtil(points, f, features=features)
    return points, features, f

def unNormalize(points, transforms, normals=None, features=[], invert=True):
    transform_functions = {
        'translation': translateUtil,
        'rotation': rotateUtil,
        'scale': reescaleUtil
    }
    if invert:
        t = {
            'translation': -transforms['translation'],
            'rotation': transforms['rotation'].T,
            'scale': 1./transforms['scale'],
            'sequence': transforms['sequence'][::-1]
        }
    else:
        t = transforms

    for key in t['sequence']:
        points, normals, features = transform_functions[key](points, t[key], normals=normals, features=features)

    return points, normals, features

def normalize(points, parameters,  normals=None, features=[]):
    transforms = {
        'translation': np.zeros(3),
        'rotation': np.eye(3),
        'scale': 1.,
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