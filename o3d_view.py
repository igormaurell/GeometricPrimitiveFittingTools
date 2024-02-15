import argparse
import open3d as o3d
import numpy as np
from tqdm import tqdm
import os
from pypcd import pypcd
import threading
from time import sleep

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

            mesh.vertices = o3d.utility.Vector3dVector(points/100000)
            geometries.append(mesh)

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
            
            elif filepath.endswith('.pcd'):
                pcd = o3d.geometry.PointCloud()

                if not args.showpcd_labels:
                    pcd_in = o3d.io.read_point_cloud(filepath, print_progress=False)
                    points = np.asarray(pcd_in.points)
                    pcd.points = o3d.utility.Vector3dVector(points)
                else:
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
    
    geometry_index = 0
    for geometry in file_geometries[geometry_index]:
        vis.add_geometry(geometry)

    def update_geometries_callback(vis):
        global geometry_index
        geometry_index += 1
        vis.clear_geometries()
        if geometry_index >= len(file_geometries):
            geometry_index = 0
        for geometry in file_geometries[geometry_index]:
            vis.add_geometry(geometry, reset_bounding_box=False)
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