import argparse

from tqdm import tqdm

from pypcd import pypcd
import open3d as o3d

import numpy as np

from shutil import rmtree
from os import listdir, makedirs
from os.path import join, isfile, exists

from lib.utils import loadFeatures, computeLabelsFromFace2Primitive, savePCD, downsampleByPointIndices, rayCastingPointCloudGeneration, funif, computeFeaturesPointIndices
from lib.writers import DatasetWriterFactory

from asGeometryOCCWrapper.surfaces import SurfaceFactory

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts a dataset from OBJ and YAML to HDF5')
    parser.add_argument('folder', type=str, help='dataset folder.')
    formats_txt = ','.join(DatasetWriterFactory.WRITERS_DICT.keys())
    parser.add_argument('formats', type=str, help=f'types of h5 format to generate. Possible formats: {formats_txt}. Multiple formats can me generated.')

    parser.add_argument('-ct', '--curve_types', type=str, default = '', help='types of curves to generate. Default = ')
    parser.add_argument('-st', '--surface_types', type=str, default = 'plane,cylinder,cone,sphere,torus,bspline', help='types of surfaces to generate. Default = plane,cylinder,cone,sphere,torus,bspline')
    parser.add_argument('-c', '--centralize', action='store_true', help='')
    parser.add_argument('-a', '--align', action='store_true', help='')
    parser.add_argument('-pnl', '--points_noise_limit', type=float, default = 0., help='')
    parser.add_argument('-nnl', '--normals_noise_limit', type=float, default = 0., help='')
    parser.add_argument('-crf', '--cube_reescale_factor', type=float, default = 0, help='')
    parser.add_argument('-no', '--normalization_order', type=str, default = 'r,c,a,pn,nn,cr', help='')

    for format in DatasetWriterFactory.WRITERS_DICT.keys():
        parser.add_argument(f'-{format}_ct', f'--{format}_curve_types', type=str, help='types of curves to generate. Default = ')
        parser.add_argument(f'-{format}_st', f'--{format}_surface_types', type=str, help='types of surfaces to generate. Default = plane,cylinder,cone,sphere')
        parser.add_argument(f'-{format}_c', f'--{format}_centralize', action='store_true', help='')
        parser.add_argument(f'-{format}_a', f'--{format}_align', action='store_true', help='')
        parser.add_argument(f'-{format}_pnl', f'--{format}_points_noise_limit', type=float, help='')
        parser.add_argument(f'-{format}_nnl', f'--{format}_normals_noise_limit', type=float, help='')
        parser.add_argument(f'-{format}_crf', f'--{format}_cube_reescale_factor', type=float, help='')
        parser.add_argument(f'-{format}_no', f'--{format}_normalization_order', type=str, default = 'r,c,a,pn,nn,cr', help='')

    parser.add_argument('--dataset_folder_name', type=str, default = 'dataset', help='dataset folder name.')
    parser.add_argument('--data_folder_name', type=str, default = 'data', help='data folder name.')
    parser.add_argument('--transform_folder_name', type=str, default = 'transform', help='transform folder name.')
    parser.add_argument('--mesh_folder_name', type=str, default = 'mesh', help='mesh folder name.')
    parser.add_argument('--features_folder_name', type=str, default = 'features', help='features folder name.')
    parser.add_argument('--pc_folder_name', type=str, default = 'pc', help='point cloud folder name.')
    parser.add_argument('-d_pc', '--delete_old_pc', action='store_true', help='')

    parser.add_argument('--pcd_generation_method', choices=['sampling', 'lidar'], type=str, default = 'sampling', help='')
    parser.add_argument('-ph', '--points_healing', action='store_true', help='')
    parser.add_argument('-nh', '--normals_healing', action='store_true', help='')
    parser.add_argument('-uon', '--use_original_noise', action='store_true', help='')
    parser.add_argument('-mps_ns', '--mesh_point_sampling_n_samples', type=int, default = 10000000, help='n_samples param for mesh_point_sampling execution, if necessary. Default: 50000000.')
    parser.add_argument('-t_p', '--train_percentage', type=float, default = 0.8, help='')
    parser.add_argument('-m_np', '--min_number_points', type=float, default = 0.0001, help='filter geometries by number of points.')
    parser.add_argument('-ls', '--leaf_size', type=float, default = 0.0, help='')

    args = vars(parser.parse_args())

    folder_name = args['folder']
    formats = [s.lower() for s in args['formats'].split(',')]
    curve_types = [s.lower() for s in args['curve_types'].split(',')]
    surface_types = [s.lower() for s in args['surface_types'].split(',')]
    centralize = args['centralize']
    align = args['align']
    points_noise_limit = args['points_noise_limit']
    normals_noise_limit = args['normals_noise_limit']
    cube_reescale_factor = args['cube_reescale_factor']
    normalization_order = args['normalization_order'].split(',')

    mps_ns = args['mesh_point_sampling_n_samples']
    delete_old_pc = args['delete_old_pc']
    train_percentage = args['train_percentage']
    use_original_noise = args['use_original_noise']
    points_healing = args['points_healing']
    normals_healing = args['normals_healing']
    min_number_points = args['min_number_points']
    min_number_points = int(min_number_points) if min_number_points > 1 else min_number_points

    pcd_generation_method = args['pcd_generation_method']
    dataset_folder_name = args['dataset_folder_name']
    data_folder_name = args['data_folder_name']
    transform_folder_name = args['transform_folder_name']
    mesh_folder_name = join(folder_name, args['mesh_folder_name'])
    features_folder_name = join(folder_name, args['features_folder_name'])
    pc_folder_name = join(folder_name, args['pc_folder_name'])
    leaf_size = args['leaf_size']

    parameters = {}
    for format in formats:
        parameters[format] = {'filter_features': {}, 'normalization': {}}

        p = args[f'{format}_curve_types']
        parameters[format]['filter_features']['curve_types'] = p if p is not None else curve_types
        p = args[f'{format}_surface_types']
        parameters[format]['filter_features']['surface_types'] = p if p is not None else surface_types
        p = args[f'{format}_centralize']
        parameters[format]['normalization']['centralize'] = p or centralize
        p = args[f'{format}_align']
        parameters[format]['normalization']['align'] = p or align
        p = args[f'{format}_points_noise_limit']
        parameters[format]['normalization']['points_noise'] = p if p is not None else points_noise_limit
        p = args[f'{format}_normals_noise_limit']
        parameters[format]['normalization']['normals_noise'] = p if p is not None else normals_noise_limit
        p = args[f'{format}_cube_reescale_factor']
        parameters[format]['normalization']['cube_rescale'] = p if p is not None else cube_reescale_factor
        p = args[f'{format}_normalization_order']
        parameters[format]['normalization']['normalization_order'] = p.split(',') if p is not None else normalization_order
        parameters[format]['train_percentage'] = train_percentage
        parameters[format]['min_number_points'] = min_number_points
        dataset_format_folder_name = join(folder_name, dataset_folder_name, format)
        parameters[format]['dataset_folder_name'] = dataset_format_folder_name
        data_format_folder_name = join(dataset_format_folder_name, data_folder_name)
        parameters[format]['data_folder_name'] = data_format_folder_name
        transform_format_folder_name = join(dataset_format_folder_name, transform_folder_name)
        parameters[format]['transform_folder_name'] = transform_format_folder_name
        if exists(dataset_format_folder_name):
            rmtree(dataset_format_folder_name)
        makedirs(dataset_format_folder_name, exist_ok=True)
        makedirs(data_format_folder_name, exist_ok=True)
        makedirs(transform_format_folder_name, exist_ok=True)

    if delete_old_pc:
        if exists(pc_folder_name):
            rmtree(pc_folder_name)
    makedirs(pc_folder_name, exist_ok=True)

    if exists(features_folder_name):
        features_files = sorted([f for f in listdir(features_folder_name) if isfile(join(features_folder_name, f))])
        print(f'\nGenerating dataset for {len(features_files)} features files...\n')
    else:
        print('\nThere is no features folder.\n')
        exit()

    if exists(mesh_folder_name):
        mesh_files, mesh_exts = zip(*[(f[:f.rfind('.')], f[f.rfind('.'):]) 
                                        for f in listdir(mesh_folder_name) if isfile(join(mesh_folder_name, f))])
    else:
        print('\nThere is no mesh folder.\n')
    
    dataset_writer_factory = DatasetWriterFactory(parameters)

    for index, features_filename in enumerate(features_files):

        point_position = features_filename.rfind('.')
        filename = features_filename[:point_position]

        print('\nGenerating Dataset - Model {} - [{}/{}]:'.format(filename, index + 1, len(features_files)))

        pc_filename = join(pc_folder_name, filename) + '.pcd'

        mesh_filename = None
        if filename in mesh_files:
            mesh_index = mesh_files.index(filename)
            mesh_filename = join(mesh_folder_name, filename) + mesh_exts[mesh_index]
    
        feature_tp =  features_filename[(point_position + 1):]

        funif(print, True)('Loading Features...')
        features_data = loadFeatures(join(features_folder_name, filename), feature_tp)
        funif(print, True)('Done.\n')

        mesh = None
        if exists(pc_filename):
            print('Opening PC...')
            pc = pypcd.PointCloud.from_path(pc_filename).pc_data
      
            points = np.vstack((pc['x'], pc['y'], pc['z'])).T
            normals = np.vstack((pc['normal_x'], pc['normal_y'], pc['normal_z'])).T
            labels_mesh = pc['label']
            
            labels, features_point_indices = computeLabelsFromFace2Primitive(labels_mesh.copy(), features_data['surfaces'])
            print('Done.\n')

        elif mesh_filename is not None:
            print('Opening Mesh...')
            mesh = o3d.io.read_triangle_mesh(mesh_filename, print_progress=True)
            print('Done.\n')

            #making mesh to clockwise (open3d default)
            mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles)[:, [1, 2, 0]])

            if pcd_generation_method == 'sampling':
                pcd, labels_mesh = mesh.sample_points_uniformly_and_trace(number_of_points=mps_ns, use_triangle_normal=True)#mesh.sample_points_uniformly(number_of_points=mps_ns, use_triangle_normal=True)
            elif pcd_generation_method == 'lidar':
                pcd, labels_mesh = rayCastingPointCloudGeneration(mesh)
            else:
                assert False, 'Point Cloud Generation Method is invalid.'

            labels_mesh = np.asarray(labels_mesh)

            points = np.asarray(pcd.points)
            normals = np.asarray(pcd.normals)

            labels, features_point_indices = computeLabelsFromFace2Primitive(labels_mesh.copy(), features_data['surfaces'])
           
            #downsample is done by surface to not mix primitives that are close to each other
            if leaf_size > 0:
                down_pcd = o3d.geometry.PointCloud()
                down_labels = []
                for i, fpi in enumerate(features_point_indices):
                    ans = downsampleByPointIndices(pcd, fpi, labels_mesh, leaf_size)
                    down_pcd += ans[0]
                    down_labels += ans[1]

                ans = downsampleByPointIndices(pcd, np.arange(len(labels))[labels==-1], labels_mesh, leaf_size)
                down_pcd += ans[0]
                down_labels += ans[1]

                perm = np.random.permutation(len(down_labels))

                points = np.asarray(down_pcd.points)[perm]
                normals = np.asarray(down_pcd.normals)[perm]
                labels_mesh = np.array(down_labels)[perm]

                labels, features_point_indices = computeLabelsFromFace2Primitive(labels_mesh.copy(), features_data['surfaces'])   

            savePCD(pc_filename,  points, normals=normals, labels=labels_mesh)

            mesh = (np.asarray(mesh.vertices), np.asarray(mesh.triangles))
        else:
            print(f'\nFeature {filename} has no PCD or OBJ to use.')
            continue
            
        if mesh is None:
            print('Opening Mesh...')
            mesh = o3d.io.read_triangle_mesh(mesh_filename, enable_post_processing=False)
            mesh = (np.asarray(mesh.vertices), np.asarray(mesh.triangles))
            print('Done.\n')

        noisy_points = None
        if points_healing or normals_healing:
            points_new = points.copy()
            normals_new = normals.copy()
            print('Healing Process...')
            for i, feature in enumerate(tqdm(features_data['surfaces'])):
                fpi = features_point_indices[i]
                if len(fpi) > 0:
                    primitive = SurfaceFactory.fromDict(feature)
                    if primitive is not None:
                        points_new[fpi], normals_new[fpi], _ = primitive.projectPointsOnGeometry(points[fpi])
            if points_healing:
                if use_original_noise:
                    noisy_points = points.copy()
                not_nan_mask = ~np.any(np.isnan(points_new), axis=1)
                points[not_nan_mask] = points_new[not_nan_mask]
            if normals_healing:
                not_nan_mask = ~np.any(np.isnan(normals_new), axis=1)
                normals[not_nan_mask] = normals_new[not_nan_mask]
            print('Done.\n')

        print('Writing...')
        dataset_writer_factory.stepAllFormats(points=points, normals=normals, labels=labels, features_data=features_data, noisy_points=noisy_points,
                                              filename=filename, features_point_indices=features_point_indices, mesh=mesh)
        print('Done.\n')

    dataset_writer_factory.finishAllFormats()