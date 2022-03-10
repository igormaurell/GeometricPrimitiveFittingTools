import numpy as np
from math import acos, pi, sqrt, tan
from lib.utils import angleVectors

# 2D

def deviationPointLine(point, normal, location, direction):
    A = location
    v = direction/np.linalg.norm(direction, ord=2)
    P = point
    n_p = normal/np.linalg.norm(normal, ord=2)
    AP = P - A
    AP_proj = np.dot(AP, v)
    return np.linalg.norm(AP - AP_proj, ord=2), abs(pi/2 - angleVectors(v, n_p))

def deviationPointCircle(point, normal, location, z_axis, radius):
    A = location
    n = z_axis/np.linalg.norm(z_axis, ord=2)
    P = point
    n_p = normal/np.linalg.norm(normal, ord=2)

    AP = P - A
    #orthogonal distance between point and circle plane
    dist_point_plane = np.dot(AP, n)

    #projection P in the circle plane and calculating the distance to the center
    P_proj = P - dist_point_plane*n
    dist_pointproj_center = np.linalg.norm(P_proj - A, ord=2)
    #if point is outside the circle arc, the distance to the curve is used 
    a = dist_pointproj_center - radius
    b = np.linalg.norm(P - P_proj, ord=2)

    angle = angleVectors(n, n_p)
    return sqrt(a**2 + b**2), angle if angle <= pi/2 else pi - angle

# 3D

def deviationPointSphere(point, normal, location, radius):
    A = location
    P = point
    n_p = normal/np.linalg.norm(normal, ord=2)
    #normal of the point projected in the sphere surface
    AP = P - A
    d = np.linalg.norm(AP, ord=2)
    #TODO: Talvez esse n_pp tambem esteja errado, mas eh menos provavel
    n_pp = AP/d
    #simple, distance from point to the center minus the sphere radius
    #angle between normals
    return abs(d - radius), angleVectors(n_pp, n_p)

def deviationPointPlane(point, normal, location, z_axis):
    A = location
    n = z_axis/np.linalg.norm(z_axis, ord=2)
    P = point
    n_p = normal/np.linalg.norm(normal, ord=2)
    AP = P - A
    d = abs(np.dot(AP, n))
    #orthogonal distance between point and plane
    #angle between normals
    angle = angleVectors(n, n_p)
    return d, angle if angle <= pi/2 else pi - angle

def deviationPointTorus(point, normal, location, z_axis, min_radius, max_radius):
    A = location
    n = z_axis
    P = point
    n_p = normal

    radius = (max_radius - min_radius)/2

    AP = P - A
    #orthogonal distance to the torus plane 
    h = np.dot(AP, n)/np.linalg.norm(n, ord = 2)

    #orthogonal distance to the revolution axis line 
    d, _ = deviationPointLine(point, normal, A, n)

    #projecting the point in the torus plane
    P_p = P - h*n/np.linalg.norm(n, ord=2)
    #getting the direction vector, using center as origin, to the point projected
    v = (P_p - A)/np.linalg.norm((P_p - A), ord=2)
    #calculating the center of circle in the direction of the input point
    B = (min_radius + radius)*v + A

    BP = P - B
    d = np.linalg.norm(BP, ord=2)
    n_pp = BP/d

    return abs(d - radius), angleVectors(n_pp, n_p)

def deviationPointCylinder(point, normal, location, z_axis, radius):
    A = location
    n = z_axis/np.linalg.norm(z_axis, ord=2)
    P = point
    n_p = normal/np.linalg.norm(normal, ord=2)
    #normal of the point projected in the sphere surface
    AP = P - A
    AP_d = np.dot(AP, n)
    P_proj = A + AP_d*n
    P_projP = P - P_proj
    d = np.linalg.norm(P_projP, ord=2)

    n_pp = P_projP/d
    #simple distance from point to the revolution axis line minus radius
    return abs(d - radius), angleVectors(n_pp, n_p)

def deviationPointCone(point, normal, location, z_axis, apex, radius, angle):
    A = location
    n = z_axis/np.linalg.norm(z_axis, ord=2)
    B = apex
    P = point
    n_p = normal/np.linalg.norm(normal, ord=2)

    AP = P - A
    AP_d = np.dot(AP, n)

    d = 0.
    angle = 0.
    if AP_d < 0:
        d, angle = deviationPointCircle(P, n_p, A, n, radius)
    else:
        BP_d = np.dot(P - B, n)
        r = BP_d*tan(angle)
        P_proj = B + BP_d*n
        PprojP = P - P_proj
        d = np.linalg.norm(PprojP, ord=2)
        #point on the surface of cone
        P_proj_surf = P_proj + r*PprojP/d
        BPprojsurf = P_proj_surf - B
        y_axis = np.cross(n, PprojP)
        n_pp = np.cross(BPprojsurf, y_axis)
        n_pp = n_pp/np.linalg.norm(n_pp, ord=2)

        d = d - r
        angle = angleVectors(n_pp, n_p)
    
    return d, angle


# def deviationNormals(n1, n2):
#     a1 = angleVectors(n1, n2)
#     a2 = angleVectors(n1, -n2)
#     if abs(a1) < abs(a2):
#         return a1
#     return a2