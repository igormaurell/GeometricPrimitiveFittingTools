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

def normalize_vectors(vectors):
    new_vectors = np.asarray([v/np.linalg.norm(v) for v in vectors])
    return new_vectors

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

def addPointsNoise(points, normals, limit=0.01):
    np.random.seed(1234)
    noise = normals * np.random.uniform(-limit, limit, (points.shape[0],1))
    points = points + noise.astype(np.float32)
    #not adding noise on normals yet
    return points

def addNormalsNoise(normals, limit=3):
    np.random.seed(1234)
    limit = np.deg2rad(limit)
    random_angles = np.random.uniform(-limit, limit, len(normals)) 

    zenith_angles = np.arccos(normals[:, 2])  # Zenith angle
    azimuth_angles = np.arctan2(normals[:, 1], normals[:, 0])  # Azimuth angle

    noisy_zenith_angles = zenith_angles + random_angles

    noisy_normals = np.column_stack((
        np.sin(noisy_zenith_angles) * np.cos(azimuth_angles),
        np.sin(noisy_zenith_angles) * np.sin(azimuth_angles),
        np.cos(noisy_zenith_angles)
    ))

    #making the noise random in a angle around the original axis (this can be simplified)
    rotate_angles = np.random.uniform(0, 2*np.pi, len(normals))
    for i, axis, angle_rad in zip(range(len(normals)), normals, rotate_angles):
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)
        ux, uy, uz = axis

        # TODO: make a function
        rotation_matrix = np.array([
            [cos_theta + ux**2 * (1 - cos_theta), ux*uy*(1 - cos_theta) - uz*sin_theta, ux*uz*(1 - cos_theta) + uy*sin_theta],
            [uy*ux*(1 - cos_theta) + uz*sin_theta, cos_theta + uy**2 * (1 - cos_theta), uy*uz*(1 - cos_theta) - ux*sin_theta],
            [uz*ux*(1 - cos_theta) - uy*sin_theta, uz*uy*(1 - cos_theta) + ux*sin_theta, cos_theta + uz**2 * (1 - cos_theta)]
        ])

        noisy_normals[i, :] = np.dot(rotation_matrix, (noisy_normals[i, :][:, np.newaxis]))[:, 0]

    return noisy_normals

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

def applyTransforms(points, transforms, normals=None, features=[], invert=True):
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

    normals = normalize_vectors(normals)

    return points, normals, features

def normalize(points, parameters, normals=None, features=[]):
    transforms = {
        'translation': np.zeros(3),
        'rotation': np.eye(3),
        'scale': 1.,
        'sequence': []
    }

    order = parameters['normalization_order'] if 'normalization_order' in parameters else ['r','c','a','pn','nn','cr']

    for n in order:
        if n=='r' and 'rescale' in parameters.keys():
            points, features, scale = rescale(points, features, parameters['rescale'])
            transforms['scale'] *= scale
            if 'scale' not in transforms['sequence']:
                transforms['sequence'].append('scale') 
        if n=='pn' and 'points_noise' in parameters.keys() and parameters['points_noise'] != 0.:
            assert normals is not None
            points = addPointsNoise(points, normals, parameters['points_noise'])
        if n=='nn' and 'normals_noise' in parameters.keys() and parameters['normals_noise'] != 0.:
            assert normals is not None
            normals = addNormalsNoise(normals, parameters['normals_noise'])
            normals = normalize_vectors(normals)
        if n=='c' and 'centralize' in parameters.keys() and parameters['centralize'] == True:
            points, features, transforms['translation'] = centralize(points, features)
            transforms['sequence'].append('translation') 
        if n=='a' and 'align' in parameters.keys() and parameters['align'] == True:
            points, normals, features, transforms['rotation'] = alignCanonical(points, normals, features)
            transforms['sequence'].append('rotation')
            normals = normalize_vectors(normals)
        if n=='cr' and 'cube_rescale' in parameters.keys() and parameters['cube_rescale'] > 0:
            points, features, scale = cubeRescale(points, features, parameters['cube_rescale'])
            transforms['scale'] *= scale
            if 'scale' not in transforms['sequence']:
                transforms['sequence'].append('scale')

    return points, normals, features, transforms

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

if __name__ == '__main__':
    SIZE = 2000
    origin_points = np.zeros((SIZE, 3))
    normals = np.zeros((SIZE, 3))
    a = np.random.rand(1,3)
    normals[:, :] = a/np.linalg.norm(a)

    normals[1:, :] = addNormalsNoise(normals[1:,:], limit=10)
    normals[0, :] *= 1.5
    
    normals*=-1

    X, Y, Z = zip(*origin_points)
    U, V, W = zip(*normals)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(X, Y, Z, U, V, W, arrow_length_ratio=0.1, pivot='tip', colors=np.concatenate((np.array([1., 0, 0])[np.newaxis, :], np.zeros((SIZE - 1, 3))), axis=0),
              linewidths=([4.0] + [0.5 for _ in range(SIZE-1)]))
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])
    plt.show()