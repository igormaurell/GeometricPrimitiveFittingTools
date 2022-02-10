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
    """
    Finds rotation matrix from vector A in 3d to vector B
    in 3d.
    B = R @ A
    """
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

def add_noise(array, limit=0.01):
    points = array[:, :3]
    normals = array[:, 3:]
    noise = normals * np.random.uniform(-limit, limit, (points.shape[0],1))
    points = points + noise.astype(np.float32)
    #not adding noise on normals yet
    noise_array = np.concatenate((points, normals), axis=1)
    return noise_array

def rotateFeatures(features, transform):
    for i in range(0, len(features)):
        for key in features[i].keys():
            if key in FEATURE_TYPE.keys():
                if FEATURE_TYPE[key] == 'point' or FEATURE_TYPE[key] == 'vector':
                    features[i][key] = list((transform @ np.array(features[i][key]).T).T)
    return features

def align_canonical(array, features=[]):
    points = array[:, :3]
    normals = array[:, 3:]
    S, U = pca_numpy(points)
    smallest_ev = U[:, np.argmin(S)]
    R = rotation_matrix_a_to_b(smallest_ev, np.array([1, 0, 0]))
    # rotate input points such that the minor principal
    # axis aligns with x axis.
    points = (R @ points.T).T
    normals= (R @ normals.T).T
    aligned_array = np.concatenate((points, normals), axis=1)
    features = rotateFeatures(features, R)
    return aligned_array, features

def translateFeatures(features, transform):
    for i in range(0, len(features)):
        for key in features[i].keys():
            if key in FEATURE_TYPE.keys():
                if FEATURE_TYPE[key] == 'point':
                    features[i][key] = list(np.array(features[i][key]) + transform)
    return features

def centralize(array, features=[]):
    points = array[:, :3]
    mean = np.mean(points, 0)
    centralized_points = points - mean
    array[:, :3] = centralized_points
    features = translateFeatures(features, -mean)
    return array, features

def reescaleFeatures(features, factor):
    for i in range(0, len(features)):
        for key in features[i].keys():
            if key in FEATURE_TYPE.keys():
                if FEATURE_TYPE[key] == 'point':
                    features[i][key] = list(np.array(features[i][key])*factor)
                if FEATURE_TYPE[key] == 'value':
                    features[i][key]*= factor
    return features

def rescale(array, features=[], factor=1000):
    points = array[:, :3]
    f = factor + EPS
    scaled_points = points*f
    array[:, :3] = scaled_points
    features = reescaleFeatures(features, f)
    return array, features

def cube_rescale(array, features=[], factor=1):
    points = array[:, :3]
    std = np.max(points, 0) - np.min(points, 0)
    f = factor/(np.max(std) + EPS)
    scaled_points = points*f
    array[:, :3] = scaled_points
    features = reescaleFeatures(features, f)
    return array, features