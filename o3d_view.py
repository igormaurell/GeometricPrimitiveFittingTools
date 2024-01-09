import argparse
import open3d as o3d
import numpy as np
from tqdm import tqdm
import os

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

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
    parser = argparse.ArgumentParser(description='Visualize point cloud data.')
    parser.add_argument('filepaths', nargs='+', type=str, help='List of paths to the point cloud files')
    parser.add_argument('--regionsizes', nargs='+', type=float, default=[0], help='Region size to generate the lineset of voxel grid')
    parser.add_argument('--meshfactor', type=float, default=1, help='Factor to scale the mesh')
    parser.add_argument('--showbbox', action='store_true', help='Show bounding box')
    parser.add_argument('--imagesfolder', type=str, default='./images', help='Folder to save the results')
    args = parser.parse_args()

    print("Load a obj point cloud, print it, and render it")
    file_geometries = []
    filenames = []
    for filepath in tqdm(args.filepaths, desc='Loading files: '):
        geometries = []

        base_filename = os.path.basename(filepath)
        base_filename = os.path.splitext(base_filename)[0]

        is_mesh = True
        mesh = o3d.io.read_triangle_mesh(filepath, print_progress=False)
        if len(mesh.triangles) == 0:
            is_mesh = False

        if is_mesh:
            points = np.asarray(mesh.vertices)/args.meshfactor
            
            bounding_box_min = np.min(points, axis=0).tolist()
            bounding_box_max = np.max(points, axis=0).tolist()
            tx = - (bounding_box_max[0] + bounding_box_min[0]) * 0.5
            ty = - (bounding_box_max[1] + bounding_box_min[1]) * 0.5
            tz = - bounding_box_min[2]
            t = np.array([tx, ty, tz])

            points += t

            mesh.vertices = o3d.utility.Vector3dVector(points)
            geometries.append(mesh)

        else:
            with open(filepath, 'r') as f:
                lines = [[float(k) for k in l[2:].split()] for l in f.readlines()]

            arr = np.asarray(lines)
            points = arr[:, :3]
            colors = arr[:, 3:]/255.
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            geometries.append(pcd)

        file_geometries.append([geometries[0]])
        filenames.append(f'{base_filename}')

        if args.showbbox:
            aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(points))
            line_set_bbox = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(aabb)
            line_set_bbox.paint_uniform_color((0.2, 0.2, 0.2))
            geometries_2 = [geometries[0], line_set_bbox]
            
            file_geometries.append(geometries_2)
            filenames.append(f'{base_filename}_bbox')
        
        for regionsize in args.regionsizes:
            if regionsize > 0:
                region_size = np.ones(3)*regionsize
                line_set_regions = comput_line_set(pcd, region_size)
                geometries_3 = [geometries[0], line_set_regions]

                file_geometries.append(geometries_3)
                filenames.append(f'{base_filename}_regionsize_{regionsize}')

    os.makedirs(args.imagesfolder, exist_ok=True)

    images_filepath = [os.path.join(args.imagesfolder, f'{filename}.png') for filename in filenames]

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    
    geometry_index = 0
    for geometry in file_geometries[geometry_index]:
        vis.add_geometry(geometry)

    def update_geometries_callback(vis):
        global geometry_index
        geometry_index += 1
        vis.clear_geometries()
        if geometry_index >= len(file_geometries):
            geometry_index = -1
        else:
            for geometry in file_geometries[geometry_index]:
                vis.add_geometry(geometry, reset_bounding_box=False)
        return True
    
    def save_image_callback(vis):
        global geometry_index
        vis.capture_screen_image(images_filepath[geometry_index])
        return True
    
    vis.register_key_callback(ord("N"), update_geometries_callback)
    vis.register_key_callback(ord("S"), save_image_callback)

    vis.run()  # user picks points

        #o3d.visualization.draw_geometries([pcd, line_set], **view_params)
    vis.destroy_window()