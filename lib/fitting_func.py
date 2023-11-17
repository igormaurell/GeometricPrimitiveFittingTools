import numpy as np
import torch
from torch.autograd import Function
import open3d as o3d
from lib.normalization import cubeRescale, rescale

torch.manual_seed(2)
np.random.seed(2)
EPS = np.finfo(np.float32).eps

read_point_cloud = o3d.io.read_point_cloud
write_point_cloud = o3d.io.write_point_cloud
PointCloud = o3d.geometry.PointCloud
Vector3dVector = o3d.utility.Vector3dVector

def write_ply(fn, point, normal=None, color=None):
  ply = PointCloud()
  ply.points = Vector3dVector(point)
  if color is not None:
    ply.colors = Vector3dVector(color)
  if normal is not None:
    ply.normals = Vector3dVector(normal)
  write_point_cloud(fn, ply)

def best_lambda(A):
    """
    Takes an under determined system and small lambda value,
    and comes up with lambda that makes the matrix A + lambda I
    invertible. Assuming A to be square matrix.
    """
    lamb = 1e-6
    cols = A.shape[0]

    for i in range(7):
        A_dash = A + lamb * torch.eye(cols)
        if cols == torch.linalg.matrix_rank(A_dash):
            # we achieved the required rank
            break
        else:
            # factor by which to increase the lambda. Choosing 10 for performance.
            lamb *= 10
    return lamb

def guard_exp(x, max_value=75, min_value=-75):
    x = torch.clamp(x, max=max_value, min=min_value)
    return torch.exp(x)

def weights_normalize(weights, bw):
    """
    Assuming that weights contains dot product of embedding of a
    points with embedding of cluster center, we want to normalize
    these weights to get probabilities. Since the clustering is
    gotten by mean shift clustering, we use the same kernel to compute
    the probabilities also.
    """
    prob = guard_exp(weights / (bw ** 2) / 2)
    prob = prob / torch.sum(prob, 0, keepdim=True)

    # This is to avoid numerical issues
    if weights.shape[0] == 1:
        return prob

    # This is done to ensure that max probability is 1 at the center.
    # this will be helpful for the spline fitting network
    prob = prob - torch.min(prob, 1, keepdim=True)[0]
    prob = prob / (torch.max(prob, 1, keepdim=True)[0] + EPS)
    return prob

def svd_grad_K(S):
    N = S.shape[0]
    s1 = S.view((1, N))
    s2 = S.view((N, 1))
    diff = s2 - s1
    plus = s2 + s1

    # TODO Look into it
    eps = torch.ones((N, N)) * 10 ** (-6)
    eps = eps.cuda(S.get_device())
    max_diff = torch.max(torch.abs(diff), eps)
    sign_diff = torch.sign(diff)

    K_neg = sign_diff * max_diff

    # gaurd the matrix inversion
    K_neg[torch.arange(N), torch.arange(N)] = 10 ** (-6)
    K_neg = 1 / K_neg
    K_pos = 1 / plus

    ones = torch.ones((N, N)).cuda(S.get_device())
    rm_diag = ones - torch.eye(N).cuda(S.get_device())
    K = K_neg * K_pos * rm_diag
    return K

def compute_grad_V(U, S, V, grad_V):
    N = S.shape[0]
    K = svd_grad_K(S)
    S = torch.eye(N).cuda(S.get_device()) * S.reshape((N, 1))
    inner = K.T * (V.T @ grad_V)
    inner = (inner + inner.T) / 2.0
    return 2 * U @ S @ inner @ V.T

class LeastSquares:
    def __init__(self):
        pass

    def lstsq(self, A, Y, lamb=0.0):
        """
        Differentiable least square
        :param A: m x n
        :param Y: n x 1
        """
        cols = A.shape[1]
        if np.isinf(A.data.cpu().numpy()).any():
            import ipdb;
            ipdb.set_trace()

        # Assuming A to be full column rank
        #print(torch.linalg.matrix_rank(A))
        if cols == torch.linalg.matrix_rank(A):
            # Full column rank
            q, r = torch.linalg.qr(A)
            x = torch.inverse(r) @ q.transpose(1, 0) @ Y
        else:
            # rank(A) < n, do regularized least square.
            AtA = A.transpose(1, 0) @ A

            # get the smallest lambda that suits our purpose, so that error in
            # results minimized.
            with torch.no_grad():
                lamb = best_lambda(AtA)
            A_dash = AtA + lamb * torch.eye(cols)
            Y_dash = A.transpose(1, 0) @ Y

            # if it still doesn't work, just set the lamb to be very high value.
            x = self.lstsq(A_dash, Y_dash, 1)
        return x

class CustomSVD(Function):
    """
    Costum SVD to deal with the situations when the
    singular values are equal. In this case, if dealt
    normally the gradient w.r.t to the input goes to inf.
    To deal with this situation, we replace the entries of
    a K matrix from eq: 13 in https://arxiv.org/pdf/1509.07838.pdf
    to high value.
    Note: only applicable for the tall and square matrix and doesn't
    give correct gradients for fat matrix. Maybe transpose of the
    original matrix is requires to deal with this situation. Left for
    future work.
    """

    @staticmethod
    def forward(ctx, input):
        # Note: input is matrix of size m x n with m >= n.
        # Note: if above assumption is voilated, the gradients
        # will be wrong.
        try:
            U, S, V = torch.svd(input, some=True)
        except:
            import ipdb;
            ipdb.set_trace()

        ctx.save_for_backward(U, S, V)
        return U, S, V

    @staticmethod
    def backward(ctx, grad_U, grad_S, grad_V):
        U, S, V = ctx.saved_tensors
        grad_input = compute_grad_V(U, S, V, grad_V)
        return grad_input

def guard_sqrt(x, minimum=1e-5):
    x = torch.clamp(x, min=minimum)
    return torch.sqrt(x)

class FittingFunctions:
    
    LS = LeastSquares()

    CUSTOMSVD = CustomSVD.apply

    @staticmethod
    def fit_plane_numpy(points, normals, weights):
        """
        Fits plane
        :param points: points with size N x 3
        :param weights: weights with size N x 1
        """
        X = points - np.sum(weights * points, 0).reshape((1, 3)) / np.sum(weights, 0)
        _, s, V = np.linalg.svd(weights * X, compute_uv=True)
        a = V.T[:, np.argmin(s)]
        a = np.reshape(a, (1, 3))
        d = np.sum(weights * (a @ points.T).T) / np.sum(weights, 0)
        return a, d

    @staticmethod
    def fit_plane_torch(points, normals, weights, show_warning=False):
        """
        Fits plane
        :param points: points with size N x 3
        :param weights: weights with size N x 1
        """
        weights_sum = torch.sum(weights) + EPS

        X = points - torch.sum(weights * points, 0).reshape((1, 3)) / weights_sum

        weighted_X = weights * X
        np_weighted_X = weighted_X.data.cpu().numpy()
        if np.linalg.cond(np_weighted_X) > 1e5:
            if show_warning:
                print("condition number is large in plane!", np.sum(np_weighted_X))
                print(torch.sum(points), torch.sum(weights))

        U, s, V = FittingFunctions.CUSTOMSVD(weighted_X)
        a = V[:, -1]
        a = torch.reshape(a, (1, 3))
        d = torch.sum(weights * (a @ points.permute(1, 0)).permute(1, 0)) / weights_sum
        return a, d

    @staticmethod
    def fit_sphere_numpy(points, normals, weights):
        dimension = points.shape[1]
        N = weights.shape[0]
        sum_weights = np.sum(weights)
        A = 2 * (- points + np.sum(points * weights, 0) / sum_weights)
        dot_points = np.sum(points * points, 1)
        normalization = np.sum(dot_points * weights) / sum_weights
        Y = dot_points - normalization
        Y = Y.reshape((N, 1))
        A = weights * A
        Y = weights * Y
        center = -np.linalg.lstsq(A, Y)[0].reshape((1, dimension))
        radius = np.sqrt(np.sum(weights[:, 0] * np.sum((points - center) ** 2, 1)) / sum_weights)
        return center, radius

    @staticmethod
    def fit_sphere_torch(points, normals, weights, show_warning=False):

        N = weights.shape[0]
        sum_weights = torch.sum(weights) + EPS
        A = 2 * (- points + torch.sum(points * weights, 0) / sum_weights)

        dot_points = weights * torch.sum(points * points, 1, keepdim=True)

        normalization = torch.sum(dot_points) / sum_weights

        Y = dot_points - normalization
        Y = Y.reshape((N, 1))
        A = weights * A
        Y = weights * Y

        if np.linalg.cond(A.data.cpu().numpy()) > 1e8:
            if show_warning:
                print("condition number is large in sphere!")

        center = -FittingFunctions.LS.lstsq(A, Y, 0.01).reshape((1, 3))
        radius_square = torch.sum(weights[:, 0] * torch.sum((points - center) ** 2, 1)) / sum_weights
        radius_square = torch.clamp(radius_square, min=1e-3)
        radius = guard_sqrt(radius_square)
        return center, radius

    @staticmethod
    def fit_cylinder_numpy(points, normals, weights):
        _, s, V = np.linalg.svd(weights * normals, compute_uv=True)
        a = V.T[:, np.argmin(s)]
        a = np.reshape(a, (1, 3))

        # find the projection onto a plane perpendicular to the axis
        a = a.reshape((3, 1))
        a = a / (np.linalg.norm(a, ord=2) + EPS)

        prj_circle = points - ((points @ a).T * a).T
        center, radius = FittingFunctions.fit_sphere_numpy(prj_circle, normals, weights)
        return a, center, radius

    @staticmethod
    def fit_cylinder_torch(points, normals, weights, show_warning=False):
        # compute
        # U, s, V = torch.svd(weights * normals)
        weighted_normals = weights * normals

        #print(np.max(points.cpu().numpy(), axis=0) - np.min(points.cpu().numpy(), axis=0))

        #print(np.linalg.cond(weighted_normals.data.cpu().numpy()) )
        if np.linalg.cond(weighted_normals.data.cpu().numpy()) > 1e5:
            if show_warning:
                print("condition number is large in cylinder")
                print(torch.sum(normals).item(), torch.sum(points).item(), torch.sum(weights).item())

        U, s, V = FittingFunctions.CUSTOMSVD(weighted_normals)
        a = V[:, -1]
        a = torch.reshape(a, (1, 3))

        # find the projection onto a plane perpendicular to the axis
        a = a.reshape((3, 1))
        a = -(a / (torch.norm(a, 2) + EPS))

        prj_circle = points - ((points @ a).permute(1, 0) * a).permute(1, 0)
        
        # torch doesn't have least square for
        center, radius = FittingFunctions.fit_sphere_torch(prj_circle, normals, weights)

        return a, center, radius

    @staticmethod
    def fit_cone_torch(points, normals, weights, show_warning=False):
        """ Need to incorporate the cholesky decomposition based
        least square fitting because it is stable and faster."""

        N = points.shape[0]
        A = weights * normals
        Y = torch.sum(normals * points, 1).reshape((N, 1))
        Y = weights * Y

        # if condition number is too large, return a very zero cone.
        if np.linalg.cond(A.data.cpu().numpy()) > 1e5:
            if show_warning:
                print("condition number is large, cone")
                print(torch.sum(normals).item(), torch.sum(points).item(), torch.sum(weights).item())
            return torch.zeros((1, 3)), torch.Tensor([[1.0, 0.0, 0.0]]), torch.zeros(1)

        c = FittingFunctions.LS.lstsq(A, Y, lamb=1e-3)

        a, _ = FittingFunctions.fit_plane_torch(normals, None, weights)
        if torch.sum(normals @ a.transpose(1, 0)) > 0:
            # we want normals to be pointing outside and axis to
            # be pointing inside the cone.
            a = - 1 * a

        diff = points - c.transpose(1, 0)
        diff = torch.nn.functional.normalize(diff, p=2, dim=1)
        diff = diff @ a.transpose(1, 0)

        # This is done to avoid the numerical issue when diff = 1 or -1
        # the derivative of acos becomes inf
        diff = torch.abs(diff)
        diff = torch.clamp(diff, max=0.999)
        theta = torch.sum(weights * torch.acos(diff)) / (torch.sum(weights) + EPS)
        theta = torch.clamp(theta, min=1e-3, max=3.142 / 2 - 1e-3)
        return c, a, theta

    FUNCTION_BY_TYPE = {
        'Plane': fit_plane_torch.__func__,
        'Cylinder': fit_cylinder_torch.__func__,
        'Cone': fit_cone_torch.__func__,
        'Sphere': fit_sphere_torch.__func__
    }

    @staticmethod
    def fit_by_global(primitive_type, points_full, normals_full, mask, weights=None):
        points_full, _, scale = cubeRescale(points_full.copy())

        points = points_full[mask]
        normals = normals_full[mask]

        feature = FittingFunctions.fit(primitive_type, points, normals, weights=weights, scale=1.0)
        _, features, _ = rescale(points, features=[feature], factor=1./scale)
        feature = features[0]

        return feature


    @staticmethod
    def fit(primitive_type, points, normals, weights=None, scale=1.0):
        if weights is None:
            weights = torch.from_numpy(np.ones([points.shape[0], 1], dtype=np.float32))
        else:
            weights = weights[:, None]
        
        points = torch.from_numpy(points)
        normals = torch.from_numpy(normals)

        result = FittingFunctions.FUNCTION_BY_TYPE[primitive_type](points, normals, weights)
        new_result = []
        for r in result:
            if isinstance(r, torch.Tensor):
                r_np = r.cpu().numpy()
            else:
                r_np = r
            r_np = r_np.flatten()
            r_np = r_np if len(r_np) > 1 else r_np[0]
            new_result.append(r_np)

        primitive_params = tuple(new_result)

        feature = FittingFunctions.params2dict(primitive_params, primitive_type)
        if feature is not None:
            feature = FittingFunctions.validate_feature(feature, scale)

        return feature

    @staticmethod
    def validate_feature(feature, scale):
        if feature['type'] == 'Cone':
            pass

        if feature['type']  == 'Cylinder':
            threshold = 10*scale
            if feature['radius'] > threshold or feature['location'][0] > threshold or \
               feature['location'][1] > threshold or feature['location'][2] > threshold:
                
                feature['invalid'] = True
        return feature

    @staticmethod
    def params2dict(params, primitive_type):
        feature = {'type': primitive_type}

        if primitive_type == 'Plane':
            z_axis, d = params

            feature['z_axis'] = z_axis.tolist()
            feature['location'] = (d*z_axis).tolist()

        elif primitive_type == 'Cone':
            location, apex, angle = params

            axis = location - apex
            dist = np.linalg.norm(axis)
            z_axis = axis/dist
            radius = np.tan(angle)*dist

            feature['angle'] = float(angle)
            feature['apex'] = apex.tolist()
            feature['location'] = location.tolist()
            feature['z_axis'] = z_axis.tolist()
            feature['radius'] = float(radius)

        elif primitive_type == 'Cylinder':
            z_axis, location, radius = params

            feature['z_axis'] = z_axis.tolist()
            feature['location'] = location.tolist()
            feature['radius'] = float(radius)

        elif primitive_type == 'Sphere':
            location, radius = params

            feature['location'] = location.tolist()
            feature['radius'] = float(radius)
        
        else:
            return None
    
        return feature