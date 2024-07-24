import argparse
import open3d as o3d
import numpy as np
from tqdm import tqdm
import os
from pypcd import pypcd
import threading
from time import sleep
from copy import deepcopy
from lib.utils import createViews, get_evenly_distributed_colors

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
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

def comput_line_set(regions, color=(0.2, 0.2, 0.2)):

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

def compute_partial_pcds(pcd, regions):
    pcds = []
    for i in range(regions.shape[0]):
        for j in range(regions.shape[1]):
            for k in range(regions.shape[2]):
                r = regions[i, j, k]
                mask = np.logical_and(np.all(pcd.points > r[0], axis=1), np.all(pcd.points < r[1], axis=1))
                pcds.append(pcd.select_by_index(np.where(mask)[0]))
    return pcds

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
def compute_partial_pcds(pcd, regions):
    pcds = []
    for i in range(regions.shape[0]):
        for j in range(regions.shape[1]):
            for k in range(regions.shape[2]):
                r = regions[i, j, k]
                mask = np.logical_and(np.all(pcd.points > r[0], axis=1), np.all(pcd.points < r[1], axis=1))
                pcds.append(pcd.select_by_index(np.where(mask)[0]))
    return pcds

def comput_extrinsic_matrix(camera_position, lookat_point, up_direction):
    # Calculate the forward direction
    forward = lookat_point - camera_position
    forward /= -np.linalg.norm(forward)

    # Calculate the right direction
    right = np.cross(up_direction, forward)
    right /= np.linalg.norm(right)

    # Calculate the new up direction
    up = np.cross(forward, right)
    up /= np.linalg.norm(up)

    # Construct the rotation matrix
    rotation_matrix = np.array([
        [right[0], right[1], right[2], 0],
        [up[0], up[1], up[2], 0],
        [-forward[0], -forward[1], -forward[2], 0],
        [0, 0, 0, 1]
    ])

    # Create the translation matrix
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = -camera_position

    # Combine rotation and translation to get the extrinsic matrix
    extrinsic_matrix = np.dot(rotation_matrix, translation_matrix)

    return extrinsic_matrix#np.linalg.inv(extrinsic_matrix)

def get_default_parameters(args):
    return {'mesh_show_wireframe': args.showmesh_wireframe}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize point cloud data.')
    parser.add_argument('filepaths', nargs='+', type=str, help='List of paths to the point cloud files')
    parser.add_argument('--regionsizes', nargs='+', type=float, default=[0], help='Region size to generate the lineset of voxel grid')
    parser.add_argument('--meshfactor', type=float, default=1, help='Factor to scale the mesh')
    parser.add_argument('--showbbox', action='store_true', help='Show bounding box')
    parser.add_argument('--imagesfolder', type=str, default='./images', help='Folder to save the results')
    parser.add_argument('--suffix', type=str, default='', help='Suffix to add to the images filenames')
    parser.add_argument('--showpcd_labels', action='store_true', help='Show point cloud labels as colors')
    parser.add_argument('--showmesh_wireframe', action='store_true', help='Show mesh wireframe')
    parser.add_argument('--showmerge_process', action='store_true', help='Show merge process')
    parser.add_argument('--showell', action='store_true', help='Show ell')
    args = parser.parse_args()
    print("Load a obj point cloud, print it, and render it")
    file_geometries = []
    filenames = []
    parameters = []
    white_mask = None
    for filepath in tqdm(args.filepaths, desc='Loading files: '):
        geometries = []
        params = []

        base_filename = os.path.basename(filepath)
        base_filename = os.path.splitext(base_filename)[0]

        is_mesh = True
        mesh = o3d.io.read_triangle_mesh(filepath, print_progress=False)
        if len(mesh.triangles) == 0:
            is_mesh = False

        if is_mesh:
          
            if filepath.endswith('.stl'):
                points = np.asarray(mesh.vertices)/args.meshfactor
            else:
                points = np.asarray(mesh.vertices)
                
            bounding_box_min = np.min(points, axis=0).tolist()
            bounding_box_max = np.max(points, axis=0).tolist()
            tx = - (bounding_box_max[0] + bounding_box_min[0]) * 0.5
            ty = - (bounding_box_max[1] + bounding_box_min[1]) * 0.5
            tz = - bounding_box_min[2]
            t = np.array([tx, ty, tz])

            points += t

            mesh.vertices = o3d.utility.Vector3dVector(points)
            geometries.append(mesh)
            param = get_default_parameters(args)
            if filepath.endswith('.stl'):
                param['mesh_show_wireframe'] = False
            params.append(param)

            if args.showell:
                dome_cell_size = 14
                distance_std = 0.
                distance = 2
                bbox = mesh.get_axis_aligned_bounding_box()
                views, dome_lines = createViews(bbox, distance=distance, cell_size=dome_cell_size, distance_std=distance_std)
                colors = np.asarray(get_evenly_distributed_colors(len(views)))/255.
                cams = []
                for i, view in enumerate(views):
                    center = mesh.get_center()
                    extrinsic = comput_extrinsic_matrix(view[:3], center, np.array([0, 0, 1]))
                    intrinsic = o3d.camera.PinholeCameraIntrinsic(
                                o3d.camera.PinholeCameraIntrinsicParameters.Kinect2ColorCameraDefault)
                    cam_curr = o3d.geometry.LineSet.create_camera_visualization(intrinsic=intrinsic,
                                                                                extrinsic=extrinsic, scale=1.0)
                    cam_curr.paint_uniform_color(colors[i])
                    cams.append(cam_curr)
                geometries.append([mesh, dome_lines] + cams)
                params.append(deepcopy(param))
        else:
            if filepath.endswith('.obj'):
                with open(filepath, 'r') as f:
                    lines = [[float(k) for k in l[2:].split()] for l in f.readlines()]

                arr = np.asarray(lines)
                points = arr[:, :3]
                colors = arr[:, 3:]/255.
                white_mask = np.all(colors == 1, axis=1)
                colors[white_mask] = [0, 0, 0] 
                
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)

                geometries.append(pcd)
                params.append(get_default_parameters(args))
            
            elif filepath.endswith('.pcd'):
                pcd = o3d.geometry.PointCloud()
                
                pcd_in = o3d.io.read_point_cloud(filepath, print_progress=False)
                points = np.asarray(pcd_in.points)
                pcd.points = o3d.utility.Vector3dVector(points)
                geometries.append(pcd)
                params.append(get_default_parameters(args))
                
                if args.showpcd_labels:
                    pc = pypcd.PointCloud.from_path(filepath).pc_data
      
                    points = np.vstack((pc['x'], pc['y'], pc['z'])).T
                    normals = np.vstack((pc['normal_x'], pc['normal_y'], pc['normal_z'])).T
                    labels = pc['label']
                    colors_table = np.random.rand(len(labels), 3)
                    colors = colors_table[labels]
                    
                    pcd.points = o3d.utility.Vector3dVector(points)
                    pcd.normals = o3d.utility.Vector3dVector(normals)
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
                regions = compute_grid(np.asarray(pcd.points), region_size)   
                line_set_regions = comput_line_set(regions[:, :])
                geometries_3 = [geometries[0], line_set_regions]

                file_geometries.append(geometries_3)
                filenames.append(f'{base_filename}_regionsize_{regionsize}')

                if args.showmerge_process:
                    merge_folder = os.path.join(args.imagesfolder, f'{base_filename}_regionsize_{regionsize}')
                    os.makedirs(merge_folder, exist_ok=True)
                    pcds = compute_partial_pcds(pcd, regions)
                    geoms = []
                    for ind, pc in enumerate(pcds):
                        geoms.append(pc)
                        file_geometries.append(geoms[0:ind+1])
                        filenames.append(os.path.join(f'{base_filename}_regionsize_{regionsize}', f'part_{ind}'))

                # regions = regions[:, :, :]*2.5
                # pcd_2 = deepcopy(pcd)
                # pcd_2 = pcd_2.scale(2.5, center=np.array([0, 0, 0]))
                # regions[:, :, :, 0, :] += region_size/3
                # regions[:, :, :, 1, :] += region_size/3
                # pcd_2 = pcd_2.translate(region_size/3, relative=True)
                # regions = regions[:, :, :]/2.5
                # pcd_2 = pcd_2.scale(1/2.5, center=np.array([0, 0, 0]))
                # line_set_regions_2 = comput_line_set(regions[:, :])
                # geometries_4 = [pcd_2, line_set_regions_2]
                # file_geometries.append(geometries_4)
                # filenames.append(f'{base_filename}_regionsize_{regionsize}_explode')                

    os.makedirs(args.imagesfolder, exist_ok=True)

    filenames_count = {}
    def get_filename_count(filename):
        if filename in filenames_count:
            filenames_count[filename] += 1
            return filenames_count[filename]
        else:
            filenames_count[filename] = 0
            return 0
        
    images_filepath = [os.path.join(args.imagesfolder, f'{filename}_{get_filename_count(filename)}{args.suffix}') for filename in filenames]
    counts = [0 for _ in filenames]

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=1080, height=1080)
    vis.get_render_option().mesh_show_wireframe = args.showmesh_wireframe
                    colors_table = np.random.rand(np.max(labels) + 1, 3)
                    colors = colors_table[labels]
                    
                    pcd2 = o3d.geometry.PointCloud()
                    pcd2.points = o3d.utility.Vector3dVector(points)
                    pcd2.normals = o3d.utility.Vector3dVector(normals)
                    pcd2.colors = o3d.utility.Vector3dVector(colors)

                    geometries.append(pcd2)
                    params.append(get_default_parameters(args))

        for i, geom in enumerate(geometries):
            if not isinstance(geom, list):
                geom = [geom]

            file_geometries.append(geom)
            parameters.append(params[i])
            filenames.append(f'{base_filename}')

            if args.showbbox:
                aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(points))
                line_set_bbox = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(aabb)
                line_set_bbox.paint_uniform_color((0.2, 0.2, 0.2))
                geometries_2 = geom + [line_set_bbox]

                file_geometries.append(geometries_2)
                parameters.append(params[i])
                filenames.append(f'{base_filename}_bbox')
            
            for regionsize in args.regionsizes:
                if regionsize > 0:
                    region_size = np.ones(3)*regionsize
                    regions = compute_grid(np.asarray(pcd.points), region_size)   
                    line_set_regions = comput_line_set(regions[:, :])
                    geometries_3 = [geom, line_set_regions]

                    file_geometries.append(geometries_3)
                    parameters.append(params[i])
                    filenames.append(f'{base_filename}_regionsize_{regionsize}')

                    if args.showmerge_process:
                        merge_folder = os.path.join(args.imagesfolder, f'{base_filename}_regionsize_{regionsize}')
                        os.makedirs(merge_folder, exist_ok=True)
                        pcds = compute_partial_pcds(pcd, regions)
                        geoms = []
                        for ind, pc in enumerate(pcds):
                            geoms.append(pc)
                            file_geometries.append(geoms[0:ind+1])
                            parameters.append(params[i])
                            filenames.append(os.path.join(f'{base_filename}_regionsize_{regionsize}', f'part_{ind}'))

                    # regions = regions[:, :, :]*2.5
                    # pcd_2 = deepcopy(pcd)
                    # pcd_2 = pcd_2.scale(2.5, center=np.array([0, 0, 0]))
                    # regions[:, :, :, 0, :] += region_size/3
                    # regions[:, :, :, 1, :] += region_size/3
                    # pcd_2 = pcd_2.translate(region_size/3, relative=True)
                    # regions = regions[:, :, :]/2.5
                    # pcd_2 = pcd_2.scale(1/2.5, center=np.array([0, 0, 0]))
                    # line_set_regions_2 = comput_line_set(regions[:, :])
                    # geometries_4 = [pcd_2, line_set_regions_2]
                    # file_geometries.append(geometries_4)
                    # filenames.append(f'{base_filename}_regionsize_{regionsize}_explode')                

    os.makedirs(args.imagesfolder, exist_ok=True)

    filenames_count = {}
    def get_filename_count(filename):
        if filename in filenames_count:
            filenames_count[filename] += 1
            return filenames_count[filename]
        else:
            filenames_count[filename] = 0
            return 0
          
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
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=1920, height=1080)
    
    geometry_index = 0
    for geometry in file_geometries[geometry_index]:
        vis.add_geometry(geometry)
        for key, value in parameters[geometry_index].items():
            setattr(vis.get_render_option(), key, value)

    def update_geometries_callback(vis):
        global geometry_index
        geometry_index += 1
        vis.clear_geometries()
        if geometry_index >= len(file_geometries):
            geometry_index = 0
        for geometry in file_geometries[geometry_index]:
            vis.add_geometry(geometry, reset_bounding_box=False)
            for key, value in parameters[geometry_index].items():
                setattr(vis.get_render_option(), key, value)
        return True
    
    def save_image_callback(vis):
        global geometry_index
        vis.capture_screen_image(f'{images_filepath[geometry_index]}_{counts[geometry_index]}.png')
        counts[geometry_index] += 1
        return True

    bursting = False    
    def burst_images_callback(vis):
        global geometry_index, bursting
        if bursting:
            return False
        initial_geometry_index = geometry_index
        bursting = True
        while True:
            save_image_callback(vis)
            if update_geometries_callback(vis):
                vis.update_renderer()
                vis.poll_events()
            if geometry_index == initial_geometry_index:
                break
        bursting = False
        return True

    video_count = 0
    def video_callback(vis):
        global video_count
        for i in range(500):
            vis.capture_screen_image(f'{images_filepath[geometry_index]}_video_{video_count}_{i}.png')
            vis.update_renderer()
            vis.poll_events()
            sleep(0.05)
        video_count += 1

    vis.register_key_callback(ord("N"), update_geometries_callback)
    vis.register_key_callback(ord("S"), save_image_callback)
    vis.register_key_callback(ord("B"), burst_images_callback)
    vis.register_key_callback(ord("V"), video_callback)

    vis.run()


    vis.destroy_window()
