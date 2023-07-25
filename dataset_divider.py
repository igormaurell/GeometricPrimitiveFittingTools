import argparse

from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map, process_map
from functools import partial

from shutil import rmtree
from os import makedirs
from os.path import join, exists

import numpy as np

import time

from lib.dataset_writer_factory import DatasetWriterFactory
from lib.dataset_reader_factory import DatasetReaderFactory

from lib.utils import writeColorPointCloudOBJ, getAllColorsArray, computeRGB
from lib.division import computeGridOfRegions, divideOnceRandom, sampleDataOnRegion, computeSearchPoints

def process_model_val(data, regions_grid, filename, val_number_points, abs_volume_threshold, relative_volume_threshold, ind):
    size_x, size_y, _, _, _ = regions_grid.shape
    k = ind // (size_y * size_x)
    j = (ind // size_x) % size_y
    i = ind % size_x

    filename_curr = f'{filename}_{i}_{j}_{k}'
    result = sampleDataOnRegion(regions_grid[i, j, k], data['points'], data['normals'], data['labels'], data['features'],
                                val_number_points, filter_features_by_volume=True, abs_volume_threshold=abs_volume_threshold,
                                relative_volume_threshold=relative_volume_threshold)
   
    result['filename'] = filename_curr
                    
    return result

def process_model_train(data, filename, train_number_points, train_min_number_points, abs_volume_threshold, relative_volume_threshold, ind):
    result = {'points': np.array([])}
    filename_curr = f'{filename}_{ind}'

    search_points = data['search_points'] if 'search_points' in data.keys() else None

    while result['points'].shape[0] < train_min_number_points:
        result = divideOnceRandom(data['points'], data['normals'], data['labels'], data['features'], train_region_size, train_number_points,
                                  filter_features_by_volume=True, abs_volume_threshold=abs_volume_threshold,
                                  relative_volume_threshold=relative_volume_threshold, search_points=search_points)
    
    result['filename'] = filename_curr

    return result

def parse_size_arg(arg, region_axis='z'):
    axis = ('x', 'y', 'z')
    assert len(arg) > 0 and len(arg) <= len(axis), 'too many argments in size arg'
    region_axis_index = axis.index(region_axis)
    size = None
    if len(arg) == 1:
        size = [float(arg[0]) for i in range(3)]
    elif len(arg) == 2:
        size = [float(arg[i]) for i in range(2)]
        size.insert(region_axis_index, float('inf'))
    else:
        size = [float(arg[i]) for i in range(3)]
    return np.asarray(size)

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
    parser.add_argument('-nl', '--noise_limit', type=float, default = 0., help='')
    parser.add_argument('-crf', '--cube_reescale_factor', type=float, default = 0, help='')

    for format in DatasetWriterFactory.WRITERS_DICT.keys():
        parser.add_argument(f'-{format}_ct', f'--{format}_curve_types', type=str, help='types of curves to generate. Default = ')
        parser.add_argument(f'-{format}_st', f'--{format}_surface_types', type=str, help='types of surfaces to generate. Default = plane,cylinder,cone,sphere')
        parser.add_argument(f'-{format}_c', f'--{format}_centralize', action='store_true', help='')
        parser.add_argument(f'-{format}_a', f'--{format}_align', action='store_true', help='')
        parser.add_argument(f'-{format}_nl', f'--{format}_noise_limit', type=float, help='')
        parser.add_argument(f'-{format}_crf', f'--{format}_cube_reescale_factor', type=float, help='')

    parser.add_argument('--input_dataset_folder_name', type=str, default = 'dataset', help='input dataset folder name.')
    parser.add_argument('--output_dataset_folder_name', type=str, default = 'dataset_divided', help='output dataset folder name.')
    parser.add_argument('--data_folder_name', type=str, default = 'data', help='data folder name.')
    parser.add_argument('--transform_folder_name', type=str, default = 'transform', help='transform folder name.')
    parser.add_argument('--pc_folder_name', type=str, default = 'pc', help='point cloud folder name.')

    parser.add_argument('-wo', '--write_obj', action='store_true', help='')
    parser.add_argument('-ra', '--region_axis', type=str, default='z', help='')
    parser.add_argument('-trs','--train_region_size', nargs='+', default=[4], help='')
    parser.add_argument('-vrs','--val_region_size', nargs='+', default=[4], help='')
    parser.add_argument('-tnp', '--train_number_points', type=int, default=0, help='')
    parser.add_argument('-vnp', '--val_number_points', type=int, default=0, help='')
    parser.add_argument('-tmnp', '--train_min_number_points', type=int, default=5000, help='')
    parser.add_argument('-vmnp', '--val_min_number_points', type=int, default=0, help='')
    parser.add_argument('-avt', '--abs_volume_threshold', type=float, default=0., help='')
    parser.add_argument('-rvt', '--relative_volume_threshold', type=float, default=0., help='')
    parser.add_argument('-tr', '--train_random_times', type=float, default=1., help='number of train regions is n = train_random_times*(model_volume/region_volume)')
    parser.add_argument('-tg', '--train_grid', action='store_true', help='flag to use division by grid in training models, it is used by default if train_random_times is 0. It is possible to use both at the same time.')
    parser.add_argument('-imnp', '--instance_min_number_points', type=float, default = 1, help='filter geometries by number of points.')

    args = vars(parser.parse_args())

    folder_name = args['folder']
    input_format = args['input_format']
    curve_types = [s.lower() for s in args['curve_types'].split(',')]
    surface_types = [s.lower() for s in args['surface_types'].split(',')]
    output_formats = [s.lower() for s in args['output_formats'].split(',')]
    centralize = args['centralize']
    align = args['align']
    noise_limit = args['noise_limit']
    cube_reescale_factor = args['cube_reescale_factor']

    input_dataset_folder_name = args['input_dataset_folder_name']
    output_dataset_folder_name = args['output_dataset_folder_name']
    data_folder_name = args['data_folder_name']
    transform_folder_name = args['transform_folder_name']
    pc_folder_name = join(folder_name, args['pc_folder_name'])

    write_obj = args['write_obj']
    region_axis = args['region_axis']
    train_region_size = parse_size_arg(args['train_region_size'], region_axis=region_axis)  
    val_region_size = parse_size_arg(args['val_region_size'], region_axis=region_axis)
    train_number_points = args['train_number_points']
    val_number_points = args['val_number_points']
    abs_volume_threshold = args['abs_volume_threshold']
    relative_volume_threshold = args['relative_volume_threshold']
    instance_min_number_points = args['instance_min_number_points']
    instance_min_number_points = int(instance_min_number_points) if instance_min_number_points >= 1 else instance_min_number_points

    train_min_number_points = args['train_min_number_points']
    val_min_number_points = args['val_min_number_points']

    train_random_times = args['train_random_times']
    train_grid = args['train_grid']

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
        p = args[f'{format}_noise_limit']
        output_parameters[format]['normalization']['add_noise'] = p if p is not None else noise_limit
        p = args[f'{format}_cube_reescale_factor']
        output_parameters[format]['min_number_points'] = instance_min_number_points
        output_parameters[format]['normalization']['cube_rescale'] = p if p is not None else cube_reescale_factor     
        output_dataset_format_folder_name = join(folder_name, output_dataset_folder_name, format)
        output_parameters[format]['dataset_folder_name'] = output_dataset_format_folder_name
        output_data_format_folder_name = join(output_dataset_format_folder_name, data_folder_name)
        output_parameters[format]['data_folder_name'] = output_data_format_folder_name
        output_transform_format_folder_name = join(output_dataset_format_folder_name, transform_folder_name)
        output_parameters[format]['transform_folder_name'] = output_transform_format_folder_name
        if exists(output_dataset_format_folder_name):
            rmtree(output_dataset_format_folder_name)
        makedirs(output_dataset_format_folder_name, exist_ok=True)
        makedirs(output_data_format_folder_name, exist_ok=True)
        makedirs(output_transform_format_folder_name, exist_ok=True)

    colors = getAllColorsArray()

    start = time.time()

    dataset_reader_factory = DatasetReaderFactory(input_parameters)

    reader = dataset_reader_factory.getReaderByFormat(input_format)

    dataset_writer_factory = DatasetWriterFactory(output_parameters)
    number_val = 0
    print('\nValidation Set:')
    reader.setCurrentSetName('val')
    dataset_writer_factory.setCurrentSetNameAllFormats('val')
    for i in range(len(reader)):
        point_cloud_full = None
        data = reader.step()
        regions_grid = computeGridOfRegions(data['points'], val_region_size)
        filename = data['filename'] if 'filename' in data.keys() else str(i)
        print('\nGenerating val dataset - Model {} - [{}/{}]:'.format(filename, i+1, len(reader)))
        full_len = np.prod(regions_grid.shape[:3])

        results = process_map(partial(process_model_val, data, regions_grid, filename, val_number_points,
                              abs_volume_threshold, relative_volume_threshold), range(full_len), chunksize=1)

        for j, result in tqdm(enumerate(results)):
            n_p = len(result['points'])
            if n_p <= val_min_number_points:
                print(f"{result['filename']} point cloud has {n_p} points. The desired amount is {val_min_number_points}")
            else:
                if write_obj:
                    points = np.zeros((result['points'].shape[0], 6))
                    points[:, 0:3] = result['points']
                    points[:, 3:6] = np.array(computeRGB(colors[j]))
                    if point_cloud_full is None:
                        point_cloud_full = points.copy()
                    else:
                        point_cloud_full = np.concatenate((point_cloud_full, points), axis=0)
                dataset_writer_factory.stepAllFormats(result['points'], normals=result['normals'], labels=result['labels'],
                                                    features_data=result['features'], filename=result['filename'])
            
        if write_obj:
            writeColorPointCloudOBJ(f'{output_data_format_folder_name}/{filename}_val.obj', point_cloud_full)

    val_end = time.time()

    print('\n\nTraining Set:')
    reader.setCurrentSetName('train')
    dataset_writer_factory.setCurrentSetNameAllFormats('train')
    train_set_len = len(reader)
    for i in range(train_set_len):
        point_cloud_full = None
        data = reader.step()
        
        filename = data['filename'] if 'filename' in data.keys() else str(i)

        print('\nGenerating training dataset - Model {} - [{}/{}]:'.format(filename, i+1, train_set_len))

        results = []
        if train_random_times == 0 or train_grid:
            regions_grid = computeGridOfRegions(data['points'], train_region_size)
            full_len = np.prod(regions_grid.shape[:3])

            results += process_map(partial(process_model_val, data, regions_grid, filename, train_number_points,
                                   abs_volume_threshold, relative_volume_threshold), range(full_len), chunksize=1)
            
        if train_random_times > 0:
            size_points = np.max(data['points'], axis=0) -  np.min(data['points'], axis=0)
            points_dim = np.prod(size_points[train_region_size != np.inf])
            num_models = int(np.ceil(train_random_times*np.ceil((points_dim/np.prod(train_region_size[train_region_size != np.inf])))))
            
            data['search_points'] = computeSearchPoints(data['points'], train_region_size)

            if num_models > 1:
                results += process_map(partial(process_model_train, data, filename, train_number_points,
                                       train_min_number_points, abs_volume_threshold, relative_volume_threshold),
                                       range(num_models), chunksize=1)
            else:
                res = sampleDataOnRegion(np.asarray((np.min(data['points'], axis=0), np.max(data['points'], axis=0))), data['points'], data['normals'],
                                            data['labels'], data['features'], train_number_points, filter_features_by_volume=True,
                                            abs_volume_threshold=abs_volume_threshold, relative_volume_threshold=relative_volume_threshold)
                res['filename'] = f'{filename}_0'
                results += [res]
       
        for j, result in tqdm(enumerate(results)):   
            n_p = len(result['points'])
            if n_p <= train_min_number_points:
                print(f"{result['filename']} point cloud has {n_p} points. The desired amount is {train_min_number_points}")
            else:
                if write_obj:
                    points = np.zeros((result['points'].shape[0], 6))
                    points[:, 0:3] = result['points']
                    points[:, 3:6] = np.array(computeRGB(colors[j]))
                    if point_cloud_full is None:
                        point_cloud_full = points.copy()
                    else:
                        point_cloud_full = np.concatenate((point_cloud_full, points), axis=0)
                dataset_writer_factory.stepAllFormats(result['points'], normals=result['normals'], labels=result['labels'],
                                                  features_data=result['features'], filename=result['filename'])

        if write_obj:
            writeColorPointCloudOBJ(f'{output_data_format_folder_name}/{filename}_train.obj', point_cloud_full)
    reader.finish()
    dataset_writer_factory.finishAllFormats()

    end = time.time()
    print('Whole proccess took: ', end - start, 's')