import argparse
from genericpath import exists

from shutil import rmtree 
from os.path import join

from lib.generate_lsspfn import generateLSSPFN

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts a dataset from OBJ and YAML to HDF5')
    parser.add_argument('folder', type=str, help='dataset folder.')
    parser.add_argument('--h5_folder_name', type=str, default = 'h5', help='h5 folder name.')
    parser.add_argument('--mesh_folder_name', type=str, default = 'mesh', help='mesh folder name.')
    parser.add_argument('--features_folder_name', type=str, default = 'features', help='features folder name.')
    parser.add_argument('--pc_folder_name', type=str, default = 'pc', help='point cloud folder name.')
    parser.add_argument('--curve_types', type=str, default = '', help='types of curves to generate. Default = ')
    parser.add_argument('--surface_types', type=str, default = 'plane,cylinder,cone,sphere', help='types of surfaces to generate. Default = plane,cylinder,cone,sphere')
    parser.add_argument('-mps_ns', '--mesh_point_sampling_n_samples', type=int, default= 50000000, help='n_samples param for mesh_point_sampling execution, if necessary. Default: 1000000.')
    parser.add_argument('-nl', '--noise_limit', type=float, default= 10, help='noise_limit for point cloud, if necessary. Default: 10.')
    parser.add_argument('-nr', '--no_regenerate', action='store_false', help='no regenerate files.')

    args = vars(parser.parse_args())

    folder_name = args['folder']
    h5_folder_name = join(folder_name, args['h5_folder_name'])
    mesh_folder_name = join(folder_name, args['mesh_folder_name'])
    features_folder_name = join(folder_name, args['features_folder_name'])
    pc_folder_name = join(folder_name, args['pc_folder_name'])
    curve_types = args['curve_types'].split(',')
    surface_types = [s.lower() for s in args['surface_types'].split(',')]
    mps_ns = str(args['mesh_point_sampling_n_samples'])
    noise_limit = args['noise_limit']
    regenerate = not args['no_regenerate']

    if regenerate:
        if exists(h5_folder_name):
            rmtree(h5_folder_name)
        if exists(pc_folder_name):
            rmtree(pc_folder_name)

    generateLSSPFN(features_folder_name, mesh_folder_name, pc_folder_name, h5_folder_name, mps_ns, noise_limit, surface_types)