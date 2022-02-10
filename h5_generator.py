import argparse
from genericpath import exists

from shutil import rmtree 
from os.path import join

from lib.generate_lsspfn import generateLSSPFN

POSSIBLE_FORMATS = ['division', 'spfn', 'parsenet']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts a dataset from OBJ and YAML to HDF5')
    parser.add_argument('folder', type=str, help='dataset folder.')
    formats_txt = ','.join(POSSIBLE_FORMATS)
    parser.add_argument('h5_formats', type=str, help=f'types of h5 format to generate. Possible formats: {formats_txt}. Multiple formats can me generated.')

    parser.add_argument('-ct', '--curve_types', type=str, default = '', help='types of curves to generate. Default = ')
    parser.add_argument('-st', '--surface_types', type=str, default = 'plane,cylinder,cone,sphere', help='types of surfaces to generate. Default = plane,cylinder,cone,sphere')
    parser.add_argument('-c', '--centralize', type=bool, default = False, help='')
    parser.add_argument('-a', '--align', type=bool, default = False, help='')
    parser.add_argument('-nl', '--noise_limit', type=float, default = 10., help='')
    parser.add_argument('-crf', '--cube_reescale_factor', type=float, default = 0, help='')

    for format in POSSIBLE_FORMATS:
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
        if format in POSSIBLE_FORMATS:
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
        parameters[format] = {}
        p = args[f'{format}_curve_types']
        parameters[format]['curve_types'] = p if p is not None else curve_types
        p = args[f'{format}_surface_types']
        parameters[format]['surface_types'] = p if p is not None else surface_types
        p = args[f'{format}_centralize']
        parameters[format]['centralize'] = p if p is not None else centralize
        p = args[f'{format}_align']
        parameters[format]['align'] = p if p is not None else align
        p = args[f'{format}_noise_limit']
        parameters[format]['noise_limit'] = p if p is not None else noise_limit
        p = args[f'{format}_cube_reescale_factor']
        parameters[format]['cube_reescale_factor'] = p if p is not None else cube_reescale_factor
        parameters[format]['mesh_point_sampling_n_samples'] = mps_ns
        parameters[format]['h5_folder_name'] = h5_folder_name
        parameters[format]['mesh_folder_name'] = mesh_folder_name
        parameters[format]['features_folder_name'] = features_folder_name
        parameters[format]['pc_folder_name'] = pc_folder_name

    if delete_old_h5:
        if exists(h5_folder_name):
            rmtree(h5_folder_name)
    if delete_old_pc:
        if exists(pc_folder_name):
            rmtree(pc_folder_name)

    generateLSSPFN(features_folder_name, mesh_folder_name, pc_folder_name, h5_folder_name, mps_ns, noise_limit, surface_types)