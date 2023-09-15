import argparse
import re

from tqdm import tqdm

import numpy as np

from os import makedirs
from os.path import join, exists
from shutil import rmtree

from lib.writers import DatasetWriterFactory
from lib.readers import DatasetReaderFactory

from lib.matching import mergeQueryAndGTData

def findLast(c, s, from_idx=0, to_idx=None):
    to_idx = len(s) if to_idx is None else to_idx

    substr = s[from_idx:to_idx]

    idx = substr.rfind(c)
    
    return idx + from_idx

def getMergedFilesDict(files):
    result = {}
    while len(files) > 0:
        file = files[0]

        u3_idx = file.rfind('_')
        u2_idx = file.rfind('_', 0, u3_idx)
        u1_idx = file.rfind('_', 0, u2_idx)

        p_idx = file.rfind('.', u3_idx, len(file))

        prefix = file[0:u1_idx]

        if p_idx > -1:
            pattern = rf"{re.escape(prefix)}_(\d+)_(\d+)_(\d+){re.escape(file[p_idx:])}"
        else:
            pattern = rf"{re.escape(prefix)}_(\d+)_(\d+)_(\d+)"

        matches = []
        new_files = []
        for query in files:
            match = re.search(pattern, query)
            if match:
                matches.append(query)
            else:
                new_files.append(query)
        
        result[prefix] = matches

        files = new_files

    return result

def addDictionaries(dict1, dict2):
    concatenate_keys = ['noisy_points', 'points', 'normals', 'labels', 'gt_indices', 'global_indices', 'non_gt_features']
    merge_keys = ['features_data']
    result_dict = dict1.copy()

    for key in dict2.keys():
        if (key in concatenate_keys or key in merge_keys) and key not in result_dict:
            result_dict[key] = dict2[key]
        elif key in concatenate_keys:
            if isinstance(result_dict[key], np.ndarray):
                result_dict[key] = np.concatenate((result_dict[key], dict2[key]))
            else:
                result_dict[key] += dict2[key]
        elif key in merge_keys:
            if isinstance(dict2[key], dict):
                # merge surfaces and curves
                pass
            else:
                i = 0
                while i < len(result_dict[key]) and i < len(dict2[key]):
                    if result_dict[key][i] is not None and dict2[key][i] is not None:
                        pass
                        # verify if is equal or merge parameters 
                    elif dict2[key][i] is not None:
                        result_dict[key][i] = dict2[key][i]
                    
                    i+= 1

                if i < len(dict2[key]):
                    result_dict[key] += dict2[key][i:]

    return result_dict
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts a dataset from OBJ and YAML to HDF5')
    parser.add_argument('folder', type=str, help='dataset folder.')
    formats_txt = ','.join(DatasetReaderFactory.READERS_DICT.keys())
    parser.add_argument('input_format', type=str, help=f'types of h5 format to generate. Possible formats: {formats_txt}. Multiple formats can me generated.')
    formats_txt = ','.join(DatasetWriterFactory.WRITERS_DICT.keys())
    parser.add_argument('output_formats', type=str, help='')

    parser.add_argument('--input_dataset_folder_name', type=str, default = 'dataset_divided', help='input dataset folder name.')
    parser.add_argument('--input_gt_dataset_folder_name', type=str, help='input dataset folder name.')
    parser.add_argument('--output_dataset_folder_name', type=str, default = 'dataset_merged', help='output dataset folder name.')
    parser.add_argument('--data_folder_name', type=str, default = 'data', help='data folder name.')
    parser.add_argument('--input_gt_data_folder_name', type=str, help='input gt data folder name.')
    parser.add_argument('--transform_folder_name', type=str, default = 'transform', help='transform folder name.')

    parser.add_argument('--use_gt_transform', action='store_true', help='flag to use transforms from ground truth dataset (not needed if the dataset folder is the same)')

    parser.add_argument('--no_use_data_primitives', action='store_true')

    args = vars(parser.parse_args())

    folder_name = args['folder']
    input_format = args['input_format']
    output_formats = [s.lower() for s in args['output_formats'].split(',')]

    input_dataset_folder_name = args['input_dataset_folder_name']
    input_gt_dataset_folder_name = args['input_gt_dataset_folder_name']
    output_dataset_folder_name = args['output_dataset_folder_name']
    data_folder_name = args['data_folder_name']
    input_gt_data_folder_name = args['input_gt_data_folder_name']
    transform_folder_name = args['transform_folder_name']

    use_gt_transform = args['use_gt_transform']

    use_data_primitives = not args['no_use_data_primitives']

    if input_gt_dataset_folder_name is not None and input_gt_data_folder_name is None:
        input_gt_data_folder_name = data_folder_name
    
    if input_gt_data_folder_name is not None and input_gt_dataset_folder_name is None:
        input_gt_dataset_folder_name = input_dataset_folder_name

    input_parameters = {}
    input_gt_parameters = {}
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

    input_gt_transform_format_folder_name = None
    if input_gt_dataset_folder_name is not None and input_gt_data_folder_name is not None:
        input_gt_parameters[input_format] = {}

        input_gt_dataset_format_folder_name = join(folder_name, input_gt_dataset_folder_name, input_format)
        input_gt_parameters[input_format]['dataset_folder_name'] = input_gt_dataset_format_folder_name
        input_gt_data_format_folder_name = join(input_gt_dataset_format_folder_name, input_gt_data_folder_name)
        input_gt_parameters[input_format]['data_folder_name'] = input_gt_data_format_folder_name
        input_gt_transform_format_folder_name = join(input_gt_dataset_format_folder_name, transform_folder_name)
        input_gt_parameters[input_format]['transform_folder_name'] = input_gt_transform_format_folder_name

    if use_gt_transform and input_gt_transform_format_folder_name is not None:
        input_parameters[input_format]['transform_folder_name'] = input_gt_transform_format_folder_name

    output_parameters = {}
    for format in output_formats:

        assert format in DatasetWriterFactory.WRITERS_DICT.keys()

        output_parameters[format] = {'filter_features': {}, 'normalization': {}}

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
    reader.setCurrentSetName('val')

    if len(input_gt_parameters) > 0:
        gt_reader_factory = DatasetReaderFactory(input_gt_parameters)
        gt_reader = gt_reader_factory.getReaderByFormat(input_format)
        gt_reader.setCurrentSetName('val')
        query_files = reader.filenames_by_set['val']
        gt_files = gt_reader.filenames_by_set['val']
        assert sorted(query_files) == sorted(gt_files), 'gt has different files from query'
    else:
        gt_reader = None

    dataset_writer_factory = DatasetWriterFactory(output_parameters)
    dataset_writer_factory.setCurrentSetNameAllFormats('val')     

    files_dict = getMergedFilesDict(reader.filenames_by_set['val'])

    print('Generating merged models...')
    for merged_filename, divided_filenames in tqdm(files_dict.items()):
        input_data = {}
        reader.filenames_by_set['val'] = sorted(divided_filenames)
        if gt_reader is not None:
            gt_reader.filenames_by_set['val'] = sorted(divided_filenames)
        global_min = -1
        num_points = 0
        gt_labels = None
        for div_filename in divided_filenames:
            data = reader.step()
            if gt_reader is not None:
                gt_data = gt_reader.step()
                if gt_labels is None:
                    gt_labels = gt_data['labels']
                else:
                    gt_labels = np.concatenate((gt_labels, gt_data['labels']))
                data = mergeQueryAndGTData(data, gt_data, global_min=global_min, num_points=num_points)
                global_min = min(global_min, np.min(data['labels']))
                num_points += len(gt_data['points'])

            input_data = addDictionaries(input_data, data)

        #adding non gt (primitives that are not in the ground truth but there are in prediction) ate the end of features list (and adjusting labels)
        if gt_reader is not None:
            input_data['features_data'] = [x for x in input_data['features_data'] if x is not None]
            num_gt_features = len(input_data['features_data'])
            input_data['features_data'] += input_data['non_gt_features']
           
            gt_labels_mask = input_data['labels'] > -1
            matching, local_labels = np.unique(input_data['labels'][gt_labels_mask], return_inverse=True)
            input_data['labels'][gt_labels_mask] = local_labels
            non_gt_labels_mask = input_data['labels'] < -1
            input_data['labels'][non_gt_labels_mask] = np.abs(input_data['labels'][non_gt_labels_mask]) + num_gt_features - 2

            gt_labels = gt_labels[input_data['gt_indices']]
            valid_gt_labels = gt_labels > -1
            gt_unique_labels = np.unique(gt_labels[valid_gt_labels])

            assert np.all(np.isin(matching, gt_unique_labels))

            matching = np.asarray([np.where(gt_unique_labels == m)[0][0] for m in matching])

            matching = np.concatenate((matching, np.zeros(len(input_data['non_gt_features']), dtype=np.int32) - 1))

            input_data['matching'] = matching
            
            del input_data['non_gt_features']

        input_data['filename'] = merged_filename

        dataset_writer_factory.stepAllFormats(**input_data)

    dataset_writer_factory.finishAllFormats()
    print('Done.')

    #print('Generating test dataset:')
    #for i in tqdm(range(len(reader))):
    #    data = reader.step()
