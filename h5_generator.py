import argparse

from tqdm import tqdm

from pypcd import pypcd

import numpy as np

from shutil import rmtree
from os import listdir, makedirs
from os.path import join, isfile, exists
from copy import deepcopy

from lib.utils import generatePCD, loadFeatures
from lib.dataset_factory import DatasetFactory

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts a dataset from OBJ and YAML to HDF5')
    parser.add_argument('folder', type=str, help='dataset folder.')
    formats_txt = ','.join(FORMATS_FUNCTION_DICT.keys())
    parser.add_argument('h5_formats', type=str, help=f'types of h5 format to generate. Possible formats: {formats_txt}. Multiple formats can me generated.')

    parser.add_argument('-ct', '--curve_types', type=str, default = '', help='types of curves to generate. Default = ')
    parser.add_argument('-st', '--surface_types', type=str, default = 'plane,cylinder,cone,sphere', help='types of surfaces to generate. Default = plane,cylinder,cone,sphere')
    parser.add_argument('-c', '--centralize', type=bool, default = False, help='')
    parser.add_argument('-a', '--align', type=bool, default = False, help='')
    parser.add_argument('-nl', '--noise_limit', type=float, default = 10., help='')
    parser.add_argument('-crf', '--cube_reescale_factor', type=float, default = 0, help='')

    for format in FORMATS_FUNCTION_DICT.keys():
        parser.add_argument(f'-{format}_ct', f'--{format}_curve_types', type=str, help='types of curves to generate. Default = ')
        parser.add_argument(f'-{format}_st', f'--{format}_surface_types', type=str, help='types of surfaces to generate. Default = plane,cylinder,cone,sphere')
        parser.add_argument(f'-{format}_c', f'--{format}_centralize', type=bool, help='')
        parser.add_argument(f'-{format}_a', f'--{format}_align', type=bool, help='')
        parser.add_argument(f'-{format}_nl', f'--{format}_noise_limit', type=float, help='')
        parser.add_argument(f'-{format}_crf', f'--{format}_cube_reescale_factor', type=float, help='')

    parser.add_argument('--h5_folder_name', type=str, default = 'h5', help='h5 folder name.')
    parser.add_argument('--mesh_folder_name', type=str, default = 'mesh', help='mesh folder name.')
    parser.add_argument('--features_folder_name', type=str, default = 'features', help='features folder name.')
    parser.add_argument('--pc_folder_name', type=str, default = 'pc', help='point cloud folder name.')

    parser.add_argument('-mps_ns', '--mesh_point_sampling_n_samples', type=int, default = 10000000, help='n_samples param for mesh_point_sampling execution, if necessary. Default: 50000000.')
    parser.add_argument('-t_p', '--train_percentage', type=int, default = 80, help='')
    parser.add_argument('-d_h5', '--delete_old_h5', action='store_true', help='')
    parser.add_argument('-d_pc', '--delete_old_pc', action='store_true', help='')

    args = vars(parser.parse_args())

    folder_name = args['folder']
    h5_formats = [s.lower() for s in args['h5_formats'].split(',')]
    aux = []
    for format in h5_formats:
        if format in FORMATS_FUNCTION_DICT.keys():
            aux.append(format)
    h5_formats = aux

    curve_types = [s.lower() for s in args['curve_types'].split(',')]
    surface_types = [s.lower() for s in args['surface_types'].split(',')]
    centralize = args['centralize']
    align = args['align']
    noise_limit = args['noise_limit']
    cube_reescale_factor = args['cube_reescale_factor']

    mps_ns = str(args['mesh_point_sampling_n_samples'])
    delete_old_h5 = args['delete_old_h5']
    delete_old_pc = args['delete_old_pc']
    train_percentage = args['train_percentage']

    h5_folder_name = args['h5_folder_name']
    mesh_folder_name = join(folder_name, args['mesh_folder_name'])
    features_folder_name = join(folder_name, args['features_folder_name'])
    pc_folder_name = join(folder_name, args['pc_folder_name'])

    parameters = {}
    for format in h5_formats:
        parameters[format] = {'filter_features': {}, 'normalization': {}}

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
        h5_format_folder_name = join(folder_name, format, h5_folder_name)
        parameters[format]['folder_name'] = h5_format_folder_name
        if delete_old_h5:
            if exists(h5_format_folder_name):
                rmtree(h5_format_folder_name)
        makedirs(h5_format_folder_name, exist_ok=True)

    if delete_old_pc:
        if exists(pc_folder_name):
            rmtree(pc_folder_name)

    if exists(features_folder_name):
        features_files = sorted([f for f in listdir(features_folder_name) if isfile(join(features_folder_name, f))])
        print(f'\nGenerating dataset for {len(features_files)} features files...\n')
    else:
        print('\nThere is no features folder.\n')
        exit()
    
    dataset_factory = DatasetFactory(parameters)

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

        dataset_factory.step(points, normals=normals, labels=labels, features_data=features_data, filename=filename)