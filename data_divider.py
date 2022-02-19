import argparse

from tqdm import tqdm

from shutil import rmtree
from os import makedirs
from os.path import join, exists

from lib.dataset_maker_factory import DatasetMakerFactory
from lib.dataset_reader_factory import DatasetReaderFactory

from lib.division import computeGridOfRegions, divideOnceRandom, sampleDataOnRegion

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts a dataset from OBJ and YAML to HDF5')
    parser.add_argument('folder', type=str, help='dataset folder.')
    formats_txt = ','.join(DatasetMakerFactory.MAKERS_DICT.keys())
    parser.add_argument('input_format', type=str, help=f'types of h5 format to generate. Possible formats: {formats_txt}. Multiple formats can me generated.')
    formats_txt = ','.join(DatasetReaderFactory.READERS_DICT.keys())
    parser.add_argument('output_formats', type=str, help='')

    parser.add_argument('-c', '--centralize', type=bool, default = True, help='')
    parser.add_argument('-a', '--align', type=bool, default = True, help='')
    parser.add_argument('-nl', '--noise_limit', type=float, default = 10., help='')
    parser.add_argument('-crf', '--cube_reescale_factor', type=float, default = 1, help='')

    for format in DatasetMakerFactory.MAKERS_DICT.keys():
        parser.add_argument(f'-{format}_c', f'--{format}_centralize', type=bool, help='')
        parser.add_argument(f'-{format}_a', f'--{format}_align', type=bool, help='')
        parser.add_argument(f'-{format}_nl', f'--{format}_noise_limit', type=float, help='')
        parser.add_argument(f'-{format}_crf', f'--{format}_cube_reescale_factor', type=float, help='')

    parser.add_argument('--input_dataset_folder_name', type=str, default = 'dataset', help='input dataset folder name.')
    parser.add_argument('--output_dataset_folder_name', type=str, default = 'dataset_divided', help='output dataset folder name.')
    parser.add_argument('--data_folder_name', type=str, default = 'data', help='data folder name.')
    parser.add_argument('--transform_folder_name', type=str, default = 'transform', help='transform folder name.')
    parser.add_argument('--mesh_folder_name', type=str, default = 'mesh', help='mesh folder name.')
    parser.add_argument('--features_folder_name', type=str, default = 'features', help='features folder name.')
    parser.add_argument('--pc_folder_name', type=str, default = 'pc', help='point cloud folder name.')
    parser.add_argument('-d_dt', '--delete_old_data', action='store_true', help='')

    parser.add_argument('-ra', '--region_axis', type=str, default='y', help='')
    parser.add_argument('-rs', '--region_size', type=str, default='4000,4000', help='')
    parser.add_argument('-np', '--number_points', type=int, default=10000, help='')
    parser.add_argument('-nt', '--number_train', type=int, default=1000, help='')
    parser.add_argument('-avt', '--abs_volume_threshold', type=float, default=0., help='')
    parser.add_argument('-rvt', '--relative_volume_threshold', type=float, default=0.2, help='')

    args = vars(parser.parse_args())

    folder_name = args['folder']
    input_format = args['input_format']
    output_formats = [s.lower() for s in args['output_formats'].split(',')]
    centralize = args['centralize']
    align = args['align']
    noise_limit = args['noise_limit']
    cube_reescale_factor = args['cube_reescale_factor']

    delete_old_data = args['delete_old_data']

    input_dataset_folder_name = args['input_dataset_folder_name']
    output_dataset_folder_name = args['output_dataset_folder_name']
    data_folder_name = args['data_folder_name']
    transform_folder_name = args['transform_folder_name']
    mesh_folder_name = join(folder_name, args['mesh_folder_name'])
    features_folder_name = join(folder_name, args['features_folder_name'])
    pc_folder_name = join(folder_name, args['pc_folder_name'])

    region_axis = args['region_axis']
    region_size = [int(s) for s in args['region_size'].split(',')]
    number_points = args['number_points']
    number_train = args['number_train']
    abs_volume_threshold = args['abs_volume_threshold']
    relative_volume_threshold = args['relative_volume_threshold']

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

        assert format in DatasetMakerFactory.MAKERS_DICT.keys()

        output_parameters[format] = {'normalization': {}}

        p = args[f'{format}_centralize']
        output_parameters[format]['normalization']['centralize'] = p if p is not None else centralize
        p = args[f'{format}_align']
        output_parameters[format]['normalization']['align'] = p if p is not None else align
        p = args[f'{format}_noise_limit']
        output_parameters[format]['normalization']['add_noise'] = p if p is not None else noise_limit
        p = args[f'{format}_cube_reescale_factor']
        output_parameters[format]['normalization']['cube_rescale'] = p if p is not None else cube_reescale_factor     
        output_dataset_format_folder_name = join(folder_name, output_dataset_folder_name, format)
        output_parameters[format]['dataset_folder_name'] = output_dataset_format_folder_name
        output_data_format_folder_name = join(output_dataset_format_folder_name, data_folder_name)
        output_parameters[format]['data_folder_name'] = output_data_format_folder_name
        output_transform_format_folder_name = join(output_dataset_format_folder_name, transform_folder_name)
        output_parameters[format]['transform_folder_name'] = output_transform_format_folder_name
        if delete_old_data:
            if exists(output_dataset_format_folder_name):
                rmtree(output_dataset_format_folder_name)
            makedirs(output_dataset_folder_name, exist_ok=True)
            makedirs(output_data_format_folder_name, exist_ok=True)
            makedirs(output_transform_format_folder_name, exist_ok=True)
    
    dataset_reader_factory = DatasetReaderFactory(input_parameters)
    reader = dataset_reader_factory.getReaderByFormat(input_format)

    dataset_maker_factory = DatasetMakerFactory(output_parameters)
    import numpy as np
    i = 0
    reader.setCurrentSetName('test')
    dataset_maker_factory.setCurrentSetNameAllFormats('test')
    while i < len(reader):
        data = reader.step()
        regions_grid = computeGridOfRegions(data['points'], region_size, region_axis)
        filename = data['filename'] if 'filename' in data.keys() else str(i)
        for j in range(len(regions_grid)):
            for k in range(len(regions_grid[i])):
                filename_curr = f'{filename}_{j}_{k}'
                result = sampleDataOnRegion(regions_grid[j][k], data['points'], data['normals'], data['labels'], data['features'], region_size, region_axis,
                                            number_points, filter_features_by_volume=True, abs_volume_threshold=abs_volume_threshold,
                                            relative_volume_threshold=relative_volume_threshold)
                n_p = result['points'].shape[0]
                if n_p < number_points:
                    print(f'Point cloud has {n_p} points. The desired amount is {number_points}')
                else:
                    dataset_maker_factory.stepAllFormats(result['points'], normals=result['normals'], labels=result['labels'],
                                                         features_data=result['features'], filename=filename_curr)
        i += 1

    i = 0
    reader.setCurrentSetName('train')
    dataset_maker_factory.setCurrentSetNameAllFormats('train')
    train_set_len = len(reader)
    div = number_train//train_set_len
    mod = number_train%train_set_len
    n_models = [div + 1 if i < mod else div for i in range(train_set_len)]
    while i < train_set_len:
        data = reader.step()
        filename = data['filename'] if 'filename' in data.keys() else str(i)
        j = 0
        while j < n_models[i]:
            filename_curr = f'{filename}_{j}'
            result = divideOnceRandom(data['points'], data['normals'], data['labels'], data['features'], region_size,
                                      region_axis, number_points, filter_features_by_volume=True, abs_volume_threshold=abs_volume_threshold,
                                      relative_volume_threshold=relative_volume_threshold)
            dataset_maker_factory.stepAllFormats(result['points'], normals=result['normals'], labels=result['labels'],
                                                 features_data=result['features'], filename=filename_curr)
            j += 1
        i += 1
    reader.finish()
    dataset_maker_factory.finishAllFormats()