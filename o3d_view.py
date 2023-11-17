import argparse

import open3d as o3d
import numpy as np
import sys

EPS = np.finfo(np.float32).eps

def getRegionAxisMinMax(region_index, axis_min, axis_max, axis_size):
    if axis_size == np.inf:
        M = axis_max
        m = axis_min
    else:
        M = axis_min + (region_index+1)*axis_size
        m = M - axis_size

    return max(m, axis_min), min(M, axis_max)

def compute_grid(points, region_size):
    min_points = np.min(points, 0)
    max_points = np.max(points, 0)
    points_size = max_points - min_points

    num_parts = np.ceil(points_size/region_size)
    num_parts = num_parts.astype('int64')
    num_parts[num_parts==0] = 1

    #adapting regions size to current model
    rs = points_size/num_parts

    min_points -= EPS
    max_points += EPS

    regions = np.ndarray((num_parts[0], num_parts[1], num_parts[2], 2, 3), dtype=np.float64)

    for i in range(num_parts[0]):
        m0, M0 = getRegionAxisMinMax(i, min_points[0], max_points[0], rs[0])
        for j in range(num_parts[1]):
            m1, M1 = getRegionAxisMinMax(j, min_points[1], max_points[1], rs[1])
            for k in range(num_parts[2]):
                m2, M2 = getRegionAxisMinMax(k, min_points[2], max_points[2], rs[2])
                regions[i, j, k, 0, :] = np.array([m0, m1, m2])
                regions[i, j, k, 1, :] = np.array([M0, M1, M2])

    return regions

def comput_line_set(pcd, region_size, color=(0.2, 0.2, 0.2)):
    regions = compute_grid(np.asarray(pcd.points), region_size)

    full_len = np.prod(regions.shape[:3])

    size_x, size_y, _, _, _ = regions.shape
    

    line_set = o3d.geometry.LineSet() 
    for ind in range(full_len):
        k = ind // (size_y * size_x)
        j = (ind // size_x) % size_y
        i = ind % size_x
        r = regions[i, j, k]

        min_vertex = r[0]
        max_vertex = r[1]

        aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_vertex, max_bound=max_vertex)

        line_set += o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(aabb)
    
    line_set.paint_uniform_color(color)

    return line_set

if __name__ == '__main__':

    REGION_SIZE = np.array([4, 4, 4])

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('filepath', type=str, help='')
    parser.add_argument('--type', choices=['pcd', 'mesh'], type=str, default='pcd', help='')

    args = vars(parser.parse_args())

    filepath = args['filepath']
    tp = args['type']

    if tp == 'pcd':
        print("Load a obj point cloud, print it, and render it")
        with open(filepath, 'r') as f:
            lines = [[float(k) for k in l[2:].split()] for l in f.readlines()]

        arr = np.asarray(lines)
        points = arr[:, :3]
        colors = arr[:, 3:]/255.

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        line_set = comput_line_set(pcd, REGION_SIZE)

        aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(pcd.points)
        
        view_data = [pcd, line_set]

    elif tp == 'mesh':
        mesh = o3d.io.read_triangle_mesh(filepath)
        vertices = np.asarray(mesh.vertices)/100000

        bounding_box_min = np.min(vertices, axis=0).tolist()
        bounding_box_max = np.max(vertices, axis=0).tolist()
        tx = - (bounding_box_max[0] + bounding_box_min[0]) * 0.5
        ty = - (bounding_box_max[1] + bounding_box_min[1]) * 0.5
        tz = - bounding_box_min[2]
        t = np.array([tx, ty, tz])

        vertices += t

        mesh.vertices = o3d.utility.Vector3dVector(vertices)

        aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(mesh.vertices)
        line_set = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(aabb)
        line_set.paint_uniform_color((0.2, 0.2, 0.2))

        view_data = [mesh, line_set]

    size = np.linalg.norm(aabb.get_max_bound() - aabb.get_min_bound())
    view_lookat = aabb.get_center()
    vertices = aabb.get_box_points()
    for v in vertices[3:7]:
        view_front = v - view_lookat
        view_front[2] = 0.7*view_front[2]
        view_front = view_front/np.linalg.norm(view_front)
        view_params = {'lookat': view_lookat, 
                    'up': np.array([0, 0, 1]), 
                    'front': view_front/np.linalg.norm(view_front),
                    'zoom': 0.035*size}

        o3d.visualization.draw_geometries(view_data, **view_params, mesh_show_wireframe=False)
