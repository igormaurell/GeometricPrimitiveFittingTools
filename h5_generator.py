import argparse

from tqdm import tqdm

from pypcd import pypcd

import numpy as np

from shutil import rmtree
from os import listdir, mkdir
from os.path import join, isfile, exists

from lib.generate_spfn import generateH52SPFN
from lib.utils import generatePCD, loadFeatures

from copy import deepcopy

from lib.utils import face2Primitive, filterFeaturesData

FORMATS_DICT = {
    'spfn': generateH52SPFN,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts a dataset from OBJ and YAML to HDF5')
    parser.add_argument('folder', type=str, help='dataset folder.')
    formats_txt = ','.join(FORMATS_DICT.keys())
    parser.add_argument('h5_formats', type=str, help=f'types of h5 format to generate. Possible formats: {formats_txt}. Multiple formats can me generated.')

    parser.add_argument('-ct', '--curve_types', type=str, default = '', help='types of curves to generate. Default = ')
    parser.add_argument('-st', '--surface_types', type=str, default = 'plane,cylinder,cone,sphere', help='types of surfaces to generate. Default = plane,cylinder,cone,sphere')
    parser.add_argument('-c', '--centralize', type=bool, default = False, help='')
    parser.add_argument('-a', '--align', type=bool, default = False, help='')
    parser.add_argument('-nl', '--noise_limit', type=float, default = 10., help='')
    parser.add_argument('-crf', '--cube_reescale_factor', type=float, default = 0, help='')

    for format in FORMATS_DICT.keys():
        parser.add_argument(f'-{format}_ct', '--{format}_curve_types', type=str, help='types of curves to generate. Default = ')
        parser.add_argument(f'-{format}_st', '--{format}_surface_types', type=str, help='types of surfaces to generate. Default = plane,cylinder,cone,sphere')
        parser.add_argument(f'-{format}_c', '--{format}_centralize', type=bool, help='')
        parser.add_argument(f'-{format}_a', '--{format}_align', type=bool, help='')
        parser.add_argument(f'-{format}_nl', '--{format}_noise_limit', type=float, help='')
        parser.add_argument(f'-{format}_crf', '--{format}_cube_reescale_factor', type=float, help='')

    parser.add_argument('--h5_folder_name', type=str, default = 'h5', help='h5 folder name.')
    parser.add_argument('--mesh_folder_name', type=str, default = 'mesh', help='mesh folder name.')
    parser.add_argument('--features_folder_name', type=str, default = 'features', help='features folder name.')
    parser.add_argument('--pc_folder_name', type=str, default = 'pc', help='point cloud folder name.')

    parser.add_argument('-mps_ns', '--mesh_point_sampling_n_samples', type=int, default= 50000000, help='n_samples param for mesh_point_sampling execution, if necessary. Default: 50000000.')
    parser.add_argument('-d_h5', '--delete_old_h5', action='store_true', help='')
    parser.add_argument('-d_pc', '--delete_old_pc', action='store_true', help='')

    args = vars(parser.parse_args())

    folder_name = args['folder']
    h5_formats = [s.lower() for s in args['h5_formats'].split(',')]
    aux = []
    for format in h5_formats:
        if format in FORMATS_DICT.keys():
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

    h5_folder_name = join(folder_name, args['h5_folder_name'])
    mesh_folder_name = join(folder_name, args['mesh_folder_name'])
    features_folder_name = join(folder_name, args['features_folder_name'])
    pc_folder_name = join(folder_name, args['pc_folder_name'])

    parameters = {}
    for format in h5_formats:
        parameters[format] = {'features': {}, 'normalization': {}}

        p = args[f'{format}_curve_types']
        parameters[format]['features']['curve_types'] = p if p is not None else curve_types
        p = args[f'{format}_surface_types']
        parameters[format]['features']['surface_types'] = p if p is not None else surface_types
        p = args[f'{format}_centralize']
        parameters[format]['normalization']['centralize'] = p if p is not None else centralize
        p = args[f'{format}_align']
        parameters[format]['normalization']['align'] = p if p is not None else align
        p = args[f'{format}_noise_limit']
        parameters[format]['normalization']['noise_limit'] = p if p is not None else noise_limit
        p = args[f'{format}_cube_reescale_factor']
        parameters[format]['normalization']['cube_reescale_factor'] = p if p is not None else cube_reescale_factor
 
    parameters_groups = []
    parameters_groups_names = []
    for format in h5_formats:
        if parameters[format]['features'] not in parameters_groups:
            parameters_groups.append(parameters[format]['features'])
            parameters_groups_names.append([format])
        else:
            index = parameters_groups.index(parameters[format]['features'])
            parameters_groups_names[index].append(format)

    if delete_old_pc:
        if exists(pc_folder_name):
            rmtree(pc_folder_name)

    if exists(features_folder_name):
        features_files = sorted([f for f in listdir(features_folder_name) if isfile(join(features_folder_name, f))])
        print(f'\nGenerating dataset for {len(features_files)} features files...\n')
    else:
        print('\nThere is no features folder.\n')
        exit()

    for features_filename in tqdm(features_files):
        print(features_filename )
        point_position = features_filename.rfind('.')
        filename = features_filename[:point_position]

        pc_filename = join(pc_folder_name, filename) + '.pcd'
        mesh_filename = join(mesh_folder_name, filename) + '.obj'
              
        if exists(pc_filename): pass
        elif exists(mesh_filename):
            if not exists(pc_folder_name):
                mkdir(pc_folder_name)
            generatePCD(pc_filename, mps_ns, mesh_filename=mesh_filename)
        else:
            print(f'\nFeature {filename} has no PCD or OBJ to use.')
            continue

        feature_tp =  features_filename[(point_position + 1):]
        features_data = loadFeatures(join(features_folder_name, filename), feature_tp)

        pc = pypcd.PointCloud.from_path(pc_filename).pc_data

        point_cloud = np.ndarray(shape=(pc['x'].shape[0], 6), dtype=np.float64)
        point_cloud[:, 0] = pc['x']
        point_cloud[:, 1] = pc['y']
        point_cloud[:, 2] = pc['z']
        point_cloud[:, 3] = pc['normal_x']
        point_cloud[:, 4] = pc['normal_y']
        point_cloud[:, 5] = pc['normal_z']

        labels = pc['label']

        for i in range(len(parameters_groups)):
            features_data_curr = deepcopy(features_data)
            filterFeaturesData(features_data_curr, parameters_groups[i]['curve_types'], parameters_groups[i]['surface_types'])
            labels_curr = deepcopy(features_data)
            face2Primitive(features_data_curr, labels_curr)

            for format in parameters_groups_names[i]:
                parameters_norm = parameters[format]['normalization']
                h5_folder_name_curr = h5_folder_name + '_' + format
                if delete_old_h5:
                    if exists(h5_folder_name_curr):
                        rmtree(h5_folder_name_curr)

                if not exists(h5_folder_name_curr):
                    mkdir(h5_folder_name_curr)

                h5_filename = join(h5_folder_name_curr, f'{filename}.h5')

                FORMATS_DICT[format](deepcopy(point_cloud), labels_curr, features_data_curr, parameters_norm, , h5_filename)
    print()