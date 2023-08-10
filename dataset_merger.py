import argparse
import re

from tqdm import tqdm

import numpy as np

from os import makedirs
from os.path import join, exists
from shutil import rmtree

from lib.dataset_writer_factory import DatasetWriterFactory
from lib.dataset_reader_factory import DatasetReaderFactory

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
    concatenate_keys = ['noisy_points', 'points', 'normals', 'labels', 'gt_indices', 'non_gt_features']
    merge_keys = ['features']
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

def mergeQueryAndGTData(query, gt, global_min=-1, num_points=0):
    if 'gt_indices' in query:
        gt_indices = np.array(query['gt_indices']) + num_points
    else:
        gt_indices = np.arange(len(query['points'])) + num_points
    
    assert 'matching' in query, 'hungarian matching not implemented yet'

    gt_labels = gt['labels']
    gt_labels_unique = np.unique(gt_labels)
    gt_labels_unique = gt_labels_unique[gt_labels_unique != -1]

    matching = np.array(query['matching'])
    valid_matching_mask = matching != -1
    matching[valid_matching_mask] = gt_labels_unique[matching[valid_matching_mask]]

    local_min = global_min
    for i in range(0, len(matching)):
        if matching[i] == -1:
            matching[i] = local_min - 1
            local_min -= 1
    
    new_labels = query['labels']
    valid_labels_mask = new_labels != -1

    new_labels[valid_labels_mask] = matching[new_labels[valid_labels_mask]]
    #print(np.unique(new_labels))

    old_features = query['features']
    new_features = [None]*(len(gt['features']))
    non_gt_features = [None]*np.count_nonzero(matching < global_min)
    for query_idx, gt_idx in enumerate(matching):
        if gt_idx < global_min:
            non_gt_features[gt_idx - global_min + 1] = old_features[query_idx]
        else:
            new_features[gt_idx] = old_features[query_idx]
    
    query['labels'] = new_labels
    query['features'] = new_features
    query['non_gt_features'] = non_gt_features
    query['gt_indices'] = gt_indices

    return query
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts a dataset from OBJ and YAML to HDF5')
    parser.add_argument('folder', type=str, help='dataset folder.')
    formats_txt = ','.join(DatasetReaderFactory.READERS_DICT.keys())
    parser.add_argument('format', type=str, help=f'types of h5 format to generate. Possible formats: {formats_txt}. Multiple formats can me generated.')

    parser.add_argument('--input_dataset_folder_name', type=str, default = 'dataset_divided', help='input dataset folder name.')
    parser.add_argument('--input_gt_dataset_folder_name', type=str, help='input dataset folder name.')
    parser.add_argument('--output_dataset_folder_name', type=str, default = 'dataset_merged', help='output dataset folder name.')
    parser.add_argument('--data_folder_name', type=str, default = 'data', help='data folder name.')
    parser.add_argument('--input_gt_data_folder_name', type=str, help='input gt data folder name.')
    parser.add_argument('--transform_folder_name', type=str, default = 'transform', help='transform folder name.')

    args = vars(parser.parse_args())

    folder_name = args['folder']
    format = args['format']

    input_dataset_folder_name = args['input_dataset_folder_name']
    input_gt_dataset_folder_name = args['input_gt_dataset_folder_name']
    output_dataset_folder_name = args['output_dataset_folder_name']
    data_folder_name = args['data_folder_name']
    input_gt_data_folder_name = args['input_gt_data_folder_name']
    transform_folder_name = args['transform_folder_name']

    if input_gt_dataset_folder_name is not None and input_gt_data_folder_name is None:
        input_gt_data_folder_name = data_folder_name
    
    if input_gt_data_folder_name is not None and input_gt_dataset_folder_name is None:
        input_gt_dataset_folder_name = input_dataset_folder_name

    input_parameters = {}
    input_gt_parameters = {}
    output_parameters = {}

    assert format in DatasetReaderFactory.READERS_DICT.keys()

    input_parameters[format] = {}

    input_dataset_format_folder_name = join(folder_name, input_dataset_folder_name, format)
    input_parameters[format]['dataset_folder_name'] = input_dataset_format_folder_name
    input_data_format_folder_name = join(input_dataset_format_folder_name, data_folder_name)
    input_parameters[format]['data_folder_name'] = input_data_format_folder_name
    input_transform_format_folder_name = join(input_dataset_format_folder_name, transform_folder_name)
    input_parameters[format]['transform_folder_name'] = input_transform_format_folder_name

    if input_gt_dataset_folder_name is not None and input_gt_data_folder_name is not None:
        input_gt_parameters[format] = {}

        input_gt_dataset_format_folder_name = join(folder_name, input_gt_dataset_folder_name, format)
        input_gt_parameters[format]['dataset_folder_name'] = input_gt_dataset_format_folder_name
        input_gt_data_format_folder_name = join(input_gt_dataset_format_folder_name, input_gt_data_folder_name)
        input_gt_parameters[format]['data_folder_name'] = input_gt_data_format_folder_name
        input_gt_transform_format_folder_name = join(input_gt_dataset_format_folder_name, transform_folder_name)
        input_gt_parameters[format]['transform_folder_name'] = input_gt_transform_format_folder_name

    output_parameters[format] = {}

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
    reader = dataset_reader_factory.getReaderByFormat(format)
    reader.setCurrentSetName('val')

    if len(input_gt_parameters) > 0:
        gt_reader_factory = DatasetReaderFactory(input_gt_parameters)
        gt_reader = gt_reader_factory.getReaderByFormat(format)
        gt_reader.setCurrentSetName('val')
        query_files = reader.filenames_by_set['val']
        gt_files = gt_reader.filenames_by_set['val']
        assert sorted(query_files) == sorted(gt_files), 'gt has different files from query'
    else:
        gt_reader = None

    dataset_writer_factory = DatasetWriterFactory(output_parameters)
    writer = dataset_writer_factory.getWriterByFormat(format)
    writer.setCurrentSetName('val')        

    files_dict = getMergedFilesDict(reader.filenames_by_set['val'])

    print('Generating merged models...')
    for merged_filename, divided_filenames in tqdm(files_dict.items()):
        input_data = {}
        reader.filenames_by_set['val'] = divided_filenames
        if gt_reader is not None:
            gt_reader.filenames_by_set['val'] = divided_filenames
        global_min = -1
        num_points = 0
        for div_filename in divided_filenames:
            data = reader.step()
            if gt_reader is not None:
                gt_data = gt_reader.step()
                data = mergeQueryAndGTData(data, gt_data, global_min=global_min, num_points=num_points)
                global_min = min(global_min, np.min(data['labels']))
                num_points += len(gt_data['points'])

            input_data = addDictionaries(input_data, data)

        #adding non gt (primitives that are not in the ground truth but there are in prediction) ate the end of features list (and adjusting labels)
        num_features = len(input_data['features'])
        input_data['features'] += input_data['non_gt_features']
        non_gt_labels_mask = input_data['labels'] < -1
        input_data['labels'][non_gt_labels_mask] = np.abs(input_data['labels'][non_gt_labels_mask]) + num_features - 2

        assert np.max(input_data['labels']) + 1 == len(input_data['features']), f"{np.max(input_data['labels']) + 1 } != {len(input_data['features'])}"
        assert np.count_nonzero(input_data['labels'] < -1) == 0, f"{np.count_nonzero(input_data['labels'] < -1)} > 0"

        input_data['features_data'] = input_data['features']
        del input_data['features']
        input_data['filename'] = merged_filename

        writer.step(**input_data)

    writer.finish()
    print('Done.')

    #print('Generating test dataset:')
    #for i in tqdm(range(len(reader))):
    #    data = reader.step()
