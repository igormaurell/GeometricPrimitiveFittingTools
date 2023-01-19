import argparse

from tqdm import tqdm

from pypcd import pypcd
import trimesh


import numpy as np

from shutil import rmtree
from os import listdir, makedirs
from os.path import join, isfile, exists

from lib.utils import generatePCD, loadFeatures, computeLabelsFromFace2Primitive
from lib.dataset_writer_factory import DatasetWriterFactory
from lib.primitive_surface_factory import PrimitiveSurfaceFactory

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts a dataset from OBJ and YAML to HDF5')
    parser.add_argument('folder', type=str, help='dataset folder.')
    formats_txt = ','.join(DatasetWriterFactory.WRITERS_DICT.keys())
    parser.add_argument('formats', type=str, help=f'types of h5 format to generate. Possible formats: {formats_txt}. Multiple formats can me generated.')

    parser.add_argument('-ct', '--curve_types', type=str, default = '', help='types of curves to generate. Default = ')
    parser.add_argument('-st', '--surface_types', type=str, default = 'plane,cylinder,cone,sphere', help='types of surfaces to generate. Default = plane,cylinder,cone,sphere')
    parser.add_argument('-c', '--centralize', type=bool, default = False, help='')
    parser.add_argument('-a', '--align', type=bool, default = False, help='')
    parser.add_argument('-nl', '--noise_limit', type=float, default = 0., help='')
    parser.add_argument('-crf', '--cube_reescale_factor', type=float, default = 0, help='')

    for format in DatasetWriterFactory.WRITERS_DICT.keys():
        parser.add_argument(f'-{format}_ct', f'--{format}_curve_types', type=str, help='types of curves to generate. Default = ')
        parser.add_argument(f'-{format}_st', f'--{format}_surface_types', type=str, help='types of surfaces to generate. Default = plane,cylinder,cone,sphere')
        parser.add_argument(f'-{format}_c', f'--{format}_centralize', type=bool, help='')
        parser.add_argument(f'-{format}_a', f'--{format}_align', type=bool, help='')
        parser.add_argument(f'-{format}_nl', f'--{format}_noise_limit', type=float, help='')
        parser.add_argument(f'-{format}_crf', f'--{format}_cube_reescale_factor', type=float, help='')

    parser.add_argument('--dataset_folder_name', type=str, default = 'dataset', help='dataset folder name.')
    parser.add_argument('--data_folder_name', type=str, default = 'data', help='data folder name.')
    parser.add_argument('--transform_folder_name', type=str, default = 'transform', help='transform folder name.')
    parser.add_argument('--mesh_folder_name', type=str, default = 'mesh', help='mesh folder name.')
    parser.add_argument('--features_folder_name', type=str, default = 'features', help='features folder name.')
    parser.add_argument('--pc_folder_name', type=str, default = 'pc', help='point cloud folder name.')
    parser.add_argument('-d_pc', '--delete_old_pc', action='store_true', help='')

    parser.add_argument('-pc', '--points_curation', action='store_true', help='')
    parser.add_argument('-nc', '--normals_curation', action='store_true', help='')
    parser.add_argument('-uon', '--use_original_noise', action='store_true', help='')
    parser.add_argument('-mps_ns', '--mesh_point_sampling_n_samples', type=int, default = 10000000, help='n_samples param for mesh_point_sampling execution, if necessary. Default: 50000000.')
    parser.add_argument('-t_p', '--train_percentage', type=int, default = 0.8, help='')
    parser.add_argument('-m_np', '--min_number_points', type=float, default = 0.0001, help='filter geometries by number of points.')

    args = vars(parser.parse_args())

    folder_name = args['folder']
    formats = [s.lower() for s in args['formats'].split(',')]
    curve_types = [s.lower() for s in args['curve_types'].split(',')]
    surface_types = [s.lower() for s in args['surface_types'].split(',')]
    centralize = args['centralize']
    align = args['align']
    noise_limit = args['noise_limit']
    cube_reescale_factor = args['cube_reescale_factor']

    mps_ns = str(args['mesh_point_sampling_n_samples'])
    delete_old_pc = args['delete_old_pc']
    train_percentage = args['train_percentage']
    use_original_noise = args['use_original_noise']
    points_curation = args['points_curation']
    normals_curation = args['normals_curation']
    min_number_points = args['min_number_points']
    min_number_points = int(min_number_points) if min_number_points > 1 else min_number_points

    dataset_folder_name = args['dataset_folder_name']
    data_folder_name = args['data_folder_name']
    transform_folder_name = args['transform_folder_name']
    mesh_folder_name = join(folder_name, args['mesh_folder_name'])
    features_folder_name = join(folder_name, args['features_folder_name'])
    pc_folder_name = join(folder_name, args['pc_folder_name'])

    parameters = {}
    for format in formats:
        parameters[format] = {'filter_features': {}, 'normalization': {}, 'input': {}}

        p = args[f'{format}_curve_types']
        parameters[format]['filter_features']['curve_types'] = p if p is not None else curve_types
        p = args[f'{format}_surface_types']
        parameters[format]['filter_features']['surface_types'] = p if p is not None else surface_types
        p = args[f'{format}_centralize']
        parameters[format]['normalization']['centralize'] = p if p is not None else centralize
        p = args[f'{format}_align']
        parameters[format]['normalization']['align'] = p if p is not None else align
        p = args[f'{format}_noise_limit']
        parameters[format]['normalization']['add_noise'] = p if p is not None else noise_limit
        p = args[f'{format}_cube_reescale_factor']
        parameters[format]['normalization']['cube_rescale'] = p if p is not None else cube_reescale_factor
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

    if exists(features_folder_name):
        features_files = sorted([f for f in listdir(features_folder_name) if isfile(join(features_folder_name, f))])
        print(f'\nGenerating dataset for {len(features_files)} features files...\n')
    else:
        print('\nThere is no features folder.\n')
        exit()
    
    dataset_writer_factory = DatasetWriterFactory(parameters)

    for features_filename in tqdm(features_files):
        point_position = features_filename.rfind('.')
        filename = features_filename[:point_position]

        pc_filename = join(pc_folder_name, filename) + '.pcd'
        mesh_filename = join(mesh_folder_name, filename) + '.obj'
              
        if exists(pc_filename): pass
        elif exists(mesh_filename):
            makedirs(pc_folder_name, exist_ok=True)
            generatePCD(pc_filename, mps_ns, mesh_filename=mesh_filename)
        else:
            print(f'\nFeature {filename} has no PCD or OBJ to use.')
            continue

        feature_tp =  features_filename[(point_position + 1):]
        features_data = loadFeatures(join(features_folder_name, filename), feature_tp)

        pc = pypcd.PointCloud.from_path(pc_filename).pc_data

        points = np.ndarray(shape=(pc['x'].shape[0], 3), dtype=np.float64)
        normals = np.ndarray(shape=(pc['normal_x'].shape[0], 3), dtype=np.float64)
        labels = np.ndarray(shape=(pc['label'].shape[0],), dtype=np.float64)
        points[:, 0] = pc['x']
        points[:, 1] = pc['y']
        points[:, 2] = pc['z']
        normals[:, 0] = pc['normal_x']
        normals[:, 1] = pc['normal_y']
        normals[:, 2] = pc['normal_z']
        labels = pc['label']

        labels, features_point_indices = computeLabelsFromFace2Primitive(labels, features_data['surfaces'])

        noisy_points = None
        if points_curation or normals_curation:
            points_new = points.copy()
            normals_new = normals.copy()
            for i, feature in enumerate(features_data['surfaces']):
                fpi = features_point_indices[i]
                if len(fpi) > 0:
                    primitive = PrimitiveSurfaceFactory.primitiveFromDict(feature)
                    if primitive is not None:
                        points_new[fpi], normals_new[fpi] = primitive.computeCorrectPointsAndNormals(points[fpi])
            if points_curation:
                if use_original_noise:
                    noisy_points = points.copy()
                points = points_new
            if normals_curation:
                normals = normals_new

        dataset_writer_factory.stepAllFormats(points, normals=normals, labels=labels, features_data=features_data, noisy_points=noisy_points, filename=filename, features_point_indices=features_point_indices)
        
    dataset_writer_factory.finishAllFormats()