import numpy as np
from math import acos, pi, sqrt

def distancePoints(A, B):
    AB = B - A
    return np.linalg.norm(AB, ord=2)

def angleVectors(n1, n2):
    c = np.dot(n1, n2)/(np.linalg.norm(n1, ord=2)*np.linalg.norm(n2, ord=2))
    c = -1.0 if c < -1 else c
    c =  1.0 if c >  1 else c
    return acos(c)

def deviationNormals(n1, n2):
    a1 = angleVectors(n1, n2)
    a2 = angleVectors(n1, -n2)
    if abs(a1) < abs(a2):
        return a1
    return a2

def deviationPointLine(point, normal, location, direction):
    A = location
    v = direction
    P = point
    n_p = normal
    AP = P - A
    return np.linalg.norm(np.cross(v, AP), ord=2)/np.linalg.norm(v, ord=2), abs(pi/2 - deviationNormals(v, n_p))

def deviationPointCircle(point, normal, curve, location, z_axis, radius):
    A = location
    n = z_axis
    P = point
    n_p = normal

    AP = P - A
    #orthogonal distance between point and circle plane
    dist_point_plane = np.dot(AP, n)/np.linalg.norm(n, ord=2)

    #projection P in the circle plane and calculating the distance to the center
    P_p = P - dist_point_plane*n/np.linalg.norm(n, ord=2)
    dist_pointproj_center = np.linalg.norm(P_p - A, ord=2)
    #if point is outside the circle arc, the distance to the curve is used 
    a = dist_pointproj_center - radius
    b = np.linalg.norm(P - P_p, ord=2)

    #calculanting tangent vector to the circle in that point
    n_pp = (P_p - A)/np.linalg.norm((P_p - A), ord=2)
    t = np.cross(n_pp, n)
    return sqrt(a**2 + b**2), abs(pi/2 - deviationNormals(t, n_p))

def deviationPointSphere(point, normal, surface, location, radius):
    A = location
    P = point
    n_p = normal
    #normal of the point projected in the sphere surface
    AP = P - A
    n_pp = AP/np.linalg.norm(AP, ord=2)
    #simple, distance from point to the center minus the sphere radius
    #angle between normals
    return abs(distancePoints(P, A) - radius), deviationNormals(n_pp, n_p)

def deviationPointPlane(point, normal, surface, location, z_axis):
    A = location
    n = z_axis
    P = point
    n_p = normal
    AP = P - A
    #orthogonal distance between point and plane
    #angle between normals
    angle = deviationNormals(n, n_p)
    if angle > pi/2:
        angle = pi - angle
    return abs(np.dot(AP, n)/np.linalg.norm(n, ord=2)), angle

def deviationPointTorus(point, normal, surface, location, z_axis, min_radius, max_radius):
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
    n_pp = BP/np.linalg.norm(BP, ord=2)

    return abs(distancePoints(B, P) - radius), deviationNormals(n_pp, n_p)

def deviationPointCylinder(point, normal, location, z_axis, radius):
    A = location
    n = z_axis
    P = point
    n_p = normal
    #normal of the point projected in the sphere surface
    AP = P - A
    n_pp = AP/np.linalg.norm(AP, ord=2)
    #simple distance from point to the revolution axis line minus radius
    return abs(deviationPointLine(point, normal, A, n)[0] - radius), deviationNormals(n_pp, n_p)

def deviationPointCone(point, normal, location, z_axis, apex, radius):
    A = location
    v = z_axis
    B = apex
    P = point
    n_p = normal

    #height of cone
    h = distancePoints(A, B)

    AP = P - A
    P_p = A + np.dot(AP, v)/np.dot(v, v) * v

    #distance from center of base to point projected
    dist_AP = distancePoints(P_p, A)
    #distance from apex to the point projected
    dist_BP = distancePoints(P_p, B)

    #if point is below the center of base, return the distance to the circle base line
    if dist_BP > dist_AP and dist_BP >= h:
        AP = P - A
        signal_dist_point_plane = np.dot(AP, v)/np.linalg.norm(v, ord=2)

        #projection P in the circle plane and calculating the distance to the center
        P_p = P - signal_dist_point_plane*v/np.linalg.norm(v, ord=2)
        dist_pointproj_center = np.linalg.norm(P_p - A, ord=2)
        #if point is outside the circle arc, the distance to the curve is used 
        if dist_pointproj_center > radius:
            #not using distance_point_circle function to not repeat operations
            a = dist_pointproj_center - radius
            b = np.linalg.norm(P - P_p, ord=2)
            n_pp = (P_p - A)/np.linalg.norm((P_p - A), ord=2)
            t = np.cross(n_pp, v)
            return sqrt(a**2 + b**2), abs(pi/2 - deviationNormals(t, n_p))
        
        #if not, the orthogonal distance to the circle plane is used
        return abs(signal_dist_point_plane), deviationNormals(-v, n_p)
    #if point is above the apex, return the distance from point to apex
    elif dist_AP > dist_BP and dist_AP >= h:
        return distancePoints(P, B), deviationNormals(v, n_p)

    #if not, calculate the radius of the circle in this point height 
    r = radius*dist_BP/h

    d = (P - P_p)/np.linalg.norm((P-P_p), ord=2)

    vr = r * d

    P_s = P_p + vr

    s = (P_s - B)/np.linalg.norm((P_s - B), ord=2)

    t = np.cross(d, v)

    n_pp = np.cross(t, s)

    #distance from point to the point projected in the revolution axis line minus the current radius
    return abs(distancePoints(P, P_p) - r), deviationNormals(n_pp, n_p)