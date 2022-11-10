import argparse

from tqdm import tqdm

from shutil import rmtree
from os import makedirs
from os.path import join, exists

import numpy as np

from lib.dataset_writer_factory import DatasetWriterFactory
from lib.dataset_reader_factory import DatasetReaderFactory

from lib.utils import writeColorPointCloudOBJ, getAllColorsArray, computeRGB
from lib.division import computeGridOfRegions, divideOnceRandom, sampleDataOnRegion
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts a dataset from OBJ and YAML to HDF5')
    parser.add_argument('folder', type=str, help='dataset folder.')
    formats_txt = ','.join(DatasetReaderFactory.READERS_DICT.keys())
    parser.add_argument('format', type=str, help=f'types of h5 format to generate. Possible formats: {formats_txt}. Multiple formats can me generated.')

    parser.add_argument('--input_dataset_folder_name', type=str, default = 'dataset', help='input dataset folder name.')
    parser.add_argument('--output_dataset_folder_name', type=str, default = 'dataset_divided', help='output dataset folder name.')
    parser.add_argument('--data_folder_name', type=str, default = 'data', help='data folder name.')
    parser.add_argument('--transform_folder_name', type=str, default = 'transform', help='transform folder name.')
    parser.add_argument('-d_dt', '--delete_old_data', action='store_true', help='')

    parser.add_argument('-wo', '--write_obj', action='store_true', help='')
    parser.add_argument('-ra', '--region_axis', type=str, default='y', help='')

    args = vars(parser.parse_args())

    folder_name = args['folder']
    format = args['format']

    delete_old_data = args['delete_old_data']

    input_dataset_folder_name = args['input_dataset_folder_name']
    output_dataset_folder_name = args['output_dataset_folder_name']
    data_folder_name = args['data_folder_name']
    transform_folder_name = args['transform_folder_name']

    write_obj = args['write_obj']
    region_axis = args['region_axis']

    input_parameters = {}
    output_parameters = {}

    assert format in DatasetReaderFactory.READERS_DICT.keys()

    input_parameters[format] = {}

    input_dataset_format_folder_name = join(folder_name, input_dataset_folder_name, format)
    input_parameters[format]['dataset_folder_name'] = input_dataset_format_folder_name
    input_data_format_folder_name = join(input_dataset_format_folder_name, data_folder_name)
    input_parameters[format]['data_folder_name'] = input_data_format_folder_name
    input_transform_format_folder_name = join(input_dataset_format_folder_name, transform_folder_name)
    input_parameters[format]['transform_folder_name'] = input_transform_format_folder_name
    
    colors = getAllColorsArray()

    dataset_reader_factory = DatasetReaderFactory(input_parameters)
    reader = dataset_reader_factory.getReaderByFormat(format)

    print('Generating test dataset:')
    for i in tqdm(range(len(reader))):
        data = reader.step()
        print(data)