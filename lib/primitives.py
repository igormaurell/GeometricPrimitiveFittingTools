"""
This defines the distance from a geometric primitive. The idea is to
sample points from the ground truth surface and find the distance of
these points from the predicted point cloud.
"""
import numpy as np
import torch
from torch.autograd.variable import Variable

EPS = np.finfo(np.float32).eps

def guard_sqrt(x, minimum=1e-5):
    x = np.clip(x, minimum, np.finfo(np.float64).max)
    return np.sqrt(x)

def chamfer_distance_single_shape(pred, gt, one_side=False, sqrt=False, reduce=True):
    """
    Computes average chamfer distance prediction and groundtruth
    :param pred: Prediction: B x N x 3
    :param gt: ground truth: B x M x 3
    :return:
    """
    if isinstance(pred, np.ndarray):
        pred = Variable(torch.from_numpy(pred.astype(np.float32))).cuda()

    if isinstance(gt, np.ndarray):
        gt = Variable(torch.from_numpy(gt.astype(np.float32))).cuda()
    pred = torch.unsqueeze(pred, 0)
    gt = torch.unsqueeze(gt, 1)

    diff = pred - gt
    diff = torch.sum(diff ** 2, 2)

    if sqrt:
        diff = guard_sqrt(diff)

    if one_side:
        cd = torch.min(diff, 1)[0]
        if reduce:
            cd = torch.mean(cd, 0)
    else:
        cd1 = torch.min(diff, 0)[0]
        cd2 = torch.min(diff, 1)[0]
        if reduce:
            cd1 = torch.mean(cd1)
            cd2 = torch.mean(cd2)
        cd = (cd1 + cd2) / 2.0
    return cd


class ResidualLoss:
    """
    Defines distance of points sampled on a patch with corresponding
    predicted patch for different primitives. There is a closed form
    formula for distance from geometric primitives, whereas for splines
    we use chamfer distance as an approximation.
    """

    def __init__(self, reduce=False, one_side=False):
        cp_distance = ComputePrimitiveDistance(reduce, one_side=one_side)
        self.routines = {"torus": cp_distance.distance_from_torus,
                         "sphere": cp_distance.distance_from_sphere,
                         "cylinder": cp_distance.distance_from_cylinder,
                         "cone": cp_distance.distance_from_cone,
                         "plane": cp_distance.distance_from_plane,
                         "bspline": cp_distance.distance_from_bspline}

    def residual_loss(self, points, parameters, sqrt=True):
        distances = self.routines[parameters['type'].lower()](points, parameters, sqrt=sqrt)

        return distances


class ComputePrimitiveDistance:
    def __init__(self, reduce=False, one_side=False):
        """
        This defines a differentiable routines that gives
        distance of a point from a surface of a predicted geometric
        primitive.
        # TODO Define closed form distance of point from bspline surface.
        """
        self.reduce = reduce
        self.one_side = one_side

    def distance_from_torus(self, points, params, sqrt=False):
        """
        Distance of points from the torus
        :param points: N x 3
        :param params: axis: 3 x 1, center: 1 x 3, major_radius \in R+, minor_radius \in R+
        """
        axis, center, major_radius, minor_radius = np.asarray(params['z_axis']), np.asarray(params['location']), params['major_radius'], params['minor_radius']
        axis = axis.reshape((3, 1)) / np.linalg.norm(axis)
        center = center.reshape((1, 3))

        center2points = points - center
        z_new = center2points @ axis  # N x 1

        x_new = guard_sqrt(np.sum(center2points ** 2, 1, keepaxis=True) - z_new ** 2)  # N x 1

        # min distance for right circle
        right_dst = (guard_sqrt((x_new - major_radius) ** 2 + z_new ** 2) - minor_radius) ** 2

        # min distance for left circle
        left_dst = (guard_sqrt((x_new + major_radius) ** 2 + z_new ** 2) - minor_radius) ** 2

        distance = np.min(right_dst, left_dst)
        distance = distance.squeeze()

        if sqrt:
            distance = guard_sqrt(distance)

        if self.reduce:
            distance = np.mean(distance)
        return distance

    def distance_from_plane(self, points, params, sqrt=False):
        """
        Distance of points from the plane
        :param points: N x 3
        :param params: a: 3 x 1, d \in R
        """
        a = np.asarray(params['z_axis'])

        d = np.dot(np.asarray(params['location']), a)/np.linalg.norm(a)

        a = a.reshape((3, 1))

        # check for the orientation
        try:
            distance = np.sum((points @ a - d) ** 2, 1)
        except:
            import ipdb;
            ipdb.set_trace()

        if sqrt:
            distance = guard_sqrt(distance)
        if self.reduce:
            distance = np.mean(distance)

        # Note that this is distance square
        return distance

    def distance_from_sphere(self, points, params, sqrt=False):
        """
        Distance of points from the sphere
        :param points: N x 3
        :param params: c: 3 x 1, radius \in R
        """
        center, radius = np.asarray(params['location']), params['radius']
        center = center.reshape((1, 3))
        distance = (np.linalg.norm(points - center, axis=1) - radius) ** 2
        if sqrt:
            distance = guard_sqrt(distance)

        if self.reduce:
            distance = np.mean(distance)
        return distance

    def distance_from_cylinder(self, points, params, sqrt=False):
        """
        Distance of points from the cylinder.
        :param points: N x 3
        :param params: axis: 3 x 1, center: 1 x 3, radius \in R
        """
        # axis: 3 x 1, center: 1 x 3
        axis, center, radius = np.asarray(params['z_axis']), np.asarray(params['location']), params['radius']
        center = center.reshape((1, 3))
        axis = axis.reshape((3, 1))

        v = points - center
        prj = (v @ axis) ** 2

        # this is going negative at some point! fix it. Numerical issues.
        # voilating pythagoras
        dist_from_surface = np.sum(v * v, 1) - prj[:, 0]
        dist_from_surface = np.clip(dist_from_surface, 1e-5, np.finfo(np.float64).max)

        distance = np.sqrt(dist_from_surface) - radius
        # distance.register_hook(self.print_norm)
        distance = distance ** 2

        length = len(distance)

        if sqrt:
            distance = guard_sqrt(distance)

        if np.sum(np.isnan(distance)):
            import ipdb;
            ipdb.set_trace()
        if self.reduce:
            distance = np.mean(distance)

        return distance

    def print_norm(self, x):
        print("printing norm 2", torch.norm(x))

    def distance_from_cone(self, points, params, sqrt=False):
        # axis: 3 x 1
        apex, axis, theta = np.asarray(params['apex']), np.asarray(params['z_axis']), params['angle']
        apex = apex.reshape((1, 3))
        axis = axis.reshape((3, 1))

        N = points.shape[0]

        # pi_2 = torch.ones(N).cuda()
        try:
            v = points - apex + 1e-8
        except:
            import ipdb;
            ipdb.set_trace()
        mod_v = np.linalg.norm(v, axis=1)
        alpha_x = (v @ axis)[:, 0] / (mod_v + 1e-7)
        alpha_x = np.clip(alpha_x, -.999, 0.999)

        # safe gaurding against arc cos derivate going at +1/-1.
        alpha = np.arccos(alpha_x)

        dist_angle = np.clip(np.abs(alpha - theta), np.finfo(np.float64).min, 3.142 / 2.0)

        distance = (mod_v * np.sin(dist_angle)) ** 2

        if sqrt:
            distance = guard_sqrt(distance)
        if self.reduce:
            distance = np.mean(distance)
        return distance

    def distance_from_bspline(self, points, params, sqrt=False):
        """
        This is a rather approximation, where we sample points on the original
        bspline surface and store it in bspline_points, and we also sample
        points on the predicted bspline surface are store them in `points`
        """
        # Need to define weighted distance.
        bspline_points = np.asarray(params['knots']) # (?) I dont know what to put here yet
        return chamfer_distance_single_shape(bspline_points, points, one_side=self.one_side, sqrt=sqrt,
                                             reduce=self.reduce)