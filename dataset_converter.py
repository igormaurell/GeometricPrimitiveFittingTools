import argparse
import re

from tqdm import tqdm

import numpy as np

from os import makedirs
from os.path import join, exists
from shutil import rmtree

from lib.writers import DatasetWriterFactory
from lib.readers import DatasetReaderFactory
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts a dataset from OBJ and YAML to HDF5')
    parser.add_argument('folder', type=str, help='dataset folder.')
    formats_txt = ','.join(DatasetReaderFactory.READERS_DICT.keys())
    parser.add_argument('input_format', type=str, help=f'types of h5 format to generate. Possible formats: {formats_txt}. Multiple formats can me generated.')
    formats_txt = ','.join(DatasetWriterFactory.WRITERS_DICT.keys())
    parser.add_argument('output_formats', type=str, help='')

    parser.add_argument('-ct', '--curve_types', type=str, default = '', help='types of curves to generate. Default = ')
    parser.add_argument('-st', '--surface_types', type=str, default = 'plane,cylinder,cone,sphere', help='types of surfaces to generate. Default = plane,cylinder,cone,sphere')
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
        parser.add_argument(f'-{format}_no', f'--{format}_normalization_order', type=str, help='')

    # TODO: add new normalization parameters for each output formats

    parser.add_argument('--input_dataset_folder_name', type=str, default = 'dataset_divided', help='input dataset folder name.')
    parser.add_argument('--output_dataset_folder_name', type=str, default = '', help='output dataset folder name.')
    parser.add_argument('--data_folder_name', type=str, default = 'data', help='data folder name.')
    parser.add_argument('--transform_folder_name', type=str, default = 'transform', help='transform folder name.')

    parser.add_argument('--no_use_data_primitives', action='store_true')

    args = vars(parser.parse_args())

    folder_name = args['folder']
    input_format = args['input_format']
    output_formats = [s.lower() for s in args['output_formats'].split(',')]
    curve_types = [s.lower() for s in args['curve_types'].split(',')]
    surface_types = [s.lower() for s in args['surface_types'].split(',')]
    output_formats = [s.lower() for s in args['output_formats'].split(',')]
    centralize = args['centralize']
    align = args['align']
    points_noise_limit = args['points_noise_limit']
    normals_noise_limit = args['normals_noise_limit']
    cube_reescale_factor = args['cube_reescale_factor']
    normalization_order = args['normalization_order'].split(',')

    input_dataset_folder_name = args['input_dataset_folder_name']
    output_dataset_folder_name = args['output_dataset_folder_name']
    output_dataset_folder_name = input_dataset_folder_name if output_dataset_folder_name == '' else output_dataset_folder_name
    data_folder_name = args['data_folder_name']
    transform_folder_name = args['transform_folder_name']

    use_data_primitives = not args['no_use_data_primitives']

    input_parameters = {}
    output_parameters = {}

    assert input_format in DatasetReaderFactory.READERS_DICT.keys()

    input_parameters[input_format] = {}

    input_dataset_format_folder_name = join(folder_name, input_dataset_folder_name, input_format)
    input_parameters[input_format]['dataset_folder_name'] = input_dataset_format_folder_name
    input_data_format_folder_name = join(input_dataset_format_folder_name, data_folder_name)
    input_parameters[input_format]['data_folder_name'] = input_data_format_folder_name
    input_transform_format_folder_name = join(input_dataset_format_folder_name, transform_folder_name)
    input_parameters[input_format]['transform_folder_name'] = input_transform_format_folder_name
    input_parameters[input_format]['use_data_primitives'] = use_data_primitives

    output_parameters = {}
    for format in output_formats:

        assert format in DatasetWriterFactory.WRITERS_DICT.keys()

        output_parameters[format] = {'filter_features': {}, 'normalization': {}}

        p = args[f'{format}_curve_types']
        output_parameters[format]['filter_features']['curve_types'] = p if p is not None else curve_types
        p = args[f'{format}_surface_types']
        output_parameters[format]['filter_features']['surface_types'] = p if p is not None else surface_types
        
        p = args[f'{format}_centralize']
        output_parameters[format]['normalization']['centralize'] = p or centralize
        p = args[f'{format}_align']
        output_parameters[format]['normalization']['align'] = p or align
        p = args[f'{format}_points_noise_limit']
        output_parameters[format]['normalization']['points_noise'] = p if p is not None else points_noise_limit
        p = args[f'{format}_normals_noise_limit']
        output_parameters[format]['normalization']['normals_noise'] = p if p is not None else normals_noise_limit
        p = args[f'{format}_cube_reescale_factor']
        output_parameters[format]['normalization']['cube_rescale'] = p if p is not None else cube_reescale_factor  
        p = args[f'{format}_normalization_order']
        output_parameters[format]['normalization']['normalization_order'] = p.split(',') if p is not None else normalization_order

        output_dataset_format_folder_name = join(folder_name, output_dataset_folder_name, format)
        output_parameters[format]['dataset_folder_name'] = output_dataset_format_folder_name
        output_data_format_folder_name = join(output_dataset_format_folder_name, data_folder_name)
        output_parameters[format]['data_folder_name'] = output_data_format_folder_name
        output_transform_format_folder_name = join(output_dataset_format_folder_name, transform_folder_name)
        output_parameters[format]['transform_folder_name'] = output_transform_format_folder_name
        makedirs(output_dataset_format_folder_name, exist_ok=True)
        if exists(output_data_format_folder_name):
            rmtree(output_data_format_folder_name)
        makedirs(output_data_format_folder_name, exist_ok=True)
        makedirs(output_transform_format_folder_name, exist_ok=True)
        
    dataset_reader_factory = DatasetReaderFactory(input_parameters)
    reader = dataset_reader_factory.getReaderByFormat(input_format)

    dataset_writer_factory = DatasetWriterFactory(output_parameters)

    print('Converting train set dataset models...')
    reader.setCurrentSetName('train')
    dataset_writer_factory.setCurrentSetNameAllFormats('train')
    for data in tqdm(reader):
        dataset_writer_factory.stepAllFormats(**data)

    print('Converting validation set dataset models...')
    reader.setCurrentSetName('val')
    dataset_writer_factory.setCurrentSetNameAllFormats('val')
    for data in tqdm(reader):
        dataset_writer_factory.stepAllFormats(**data)

    dataset_writer_factory.finishAllFormats()
    print('Done.')