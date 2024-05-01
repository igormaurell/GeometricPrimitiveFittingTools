from OCC.Core.Geom import Geom_CylindricalSurface, Geom_ConicalSurface, Geom_Plane, Geom_SphericalSurface, Geom_BSplineSurface, Geom_SurfaceOfLinearExtrusion, Geom_SurfaceOfRevolution, Geom_ToroidalSurface
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCC.Core.gp import gp_Ax3, gp_Pnt, gp_Dir
from OCC.Core.TColgp import TColgp_Array2OfPnt
from OCC.Core.TColStd import TColStd_Array2OfReal, TColStd_Array1OfReal, TColStd_Array1OfInteger

import numpy as np

class SurfacesProjector:
    PROJECTOR = GeomAPI_ProjectPointOnSurf()

    def buildCylinder(features):
        radius = features['radius']
        location = features['location']
        z_axis = features['z_axis']
        axis = gp_Ax3(gp_Pnt(location[0], location[1], location[2]), gp_Dir(z_axis[0], z_axis[1], z_axis[2]))
        surf = Geom_CylindricalSurface(axis, radius)
        return surf

    def buildBSpline(features):
        poles = np.asarray(features['poles'])
        poles_tcol = TColgp_Array2OfPnt(1, poles.shape[0], 1, poles.shape[1])
        for i in range(poles.shape[0]):
            for j in range(poles.shape[1]):
                print(i, j)
                e = poles[i, j, :]
                #poles_tcol.SetValue(i, j, gp_Pnt(e[0], e[1], e[2]) )

        weights = features['weights']
        u_knots = features['u_knots']
        v_knots = features['v_knots']
        u_degree = features['u_degree']
        v_degree = features['v_degree']
        surf = Geom_BSplineSurface(poles_tcol, TColStd_Array2OfReal(weights), TColStd_Array1OfReal(u_knots), TColStd_Array1OfReal(v_knots), TColStd_Array1OfInteger([1 for i in range(len(u_knots))]), TColStd_Array1OfInteger([1 for i in range(len(v_knots))]), u_degree, v_degree)
        return surf

    def buildCone(features):
        radius = features['radius']
        location = features['location']
        z_axis = features['z_axis']
        angle = features['angle']
        axis = gp_Ax3(gp_Pnt(location[0], location[1], location[2]), gp_Dir(z_axis[0], z_axis[1], z_axis[2]))
        surf = Geom_ConicalSurface(axis, angle, radius)
        return surf

    def buildPlane(features):
        location = features['location']
        normal = features['z_axis']
        axis = gp_Ax3(gp_Pnt(location[0], location[1], location[2]), gp_Dir(normal[0], normal[1], normal[2]))
        surf = Geom_Plane(axis)
        return surf

    def buildSphere(features):
        radius = features['radius']
        location = features['location']
        z_axis = features['z_axis']
        axis = gp_Ax3(gp_Pnt(location[0], location[1], location[2]), gp_Dir(z_axis[0], z_axis[1], z_axis[2]))
        surf = Geom_SphericalSurface(axis, radius)
        return surf

    def buildSurfaceRevolution(features):
        pass

    def buildTorus(features):
        min_radius = features['min_radius']
        max_radius = features['max_radius']
        location = features['location']
        z_axis = features['z_axis']
        axis = gp_Ax3(gp_Pnt(location[0], location[1], location[2]), gp_Dir(z_axis[0], z_axis[1], z_axis[2]))
        surf = Geom_ToroidalSurface(axis, max_radius, min_radius)
        return surf

    def buildSurfaceExtrusion(features):
        pass

    BUILD_FUNCTION_MAP = {
        'cylinder': buildCylinder,
        'cone': buildCone,
        'plane': buildPlane,
        'sphere': buildSphere,
        'torus': buildTorus
    }

    @staticmethod
    def projectPointsOnSurfaceFeatures(points, features):
        tp = features['type'].lower()
        if tp in SurfacesProjector.BUILD_FUNCTION_MAP.keys():
            if len(points) == 0:
                return [], []
            surface = SurfacesProjector.BUILD_FUNCTION_MAP[tp](features)
            uvs = np.zeros((points.shape[0], 2), dtype=np.float64)
            points_projected = np.zeros(points.shape, dtype=np.float64)
            SurfacesProjector.PROJECTOR.Init(gp_Pnt(points[0, 0], points[0, 1], points[0, 2]), surface)
            uvs[0] = np.array(SurfacesProjector.PROJECTOR.LowerDistanceParameters())
            pnt = SurfacesProjector.PROJECTOR.NearestPoint()
            points_projected[0] = np.array([pnt.X(), pnt.Y(), pnt.Z()])
            for idx, pt in enumerate(points[1:, :]):
                SurfacesProjector.PROJECTOR.Perform(gp_Pnt(pt[0], pt[1], pt[2]))
                uvs[idx+1] = np.array(SurfacesProjector.PROJECTOR.LowerDistanceParameters())
                pnt = SurfacesProjector.PROJECTOR.NearestPoint()
                points_projected[idx+1] = np.array([pnt.X(), pnt.Y(), pnt.Z()])
            return uvs, points_projected
        else:
            return None