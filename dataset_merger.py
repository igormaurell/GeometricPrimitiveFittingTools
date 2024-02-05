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
        if key in concatenate_keys:
            if key not in result_dict:
                result_dict[key] = dict2[key]
            elif isinstance(result_dict[key], np.ndarray):
                result_dict[key] = np.concatenate((result_dict[key], dict2[key]))
            else:
                result_dict[key] += dict2[key]
        elif key in merge_keys:
            if key not in result_dict:
                i = 0
                result_dict[key] = []
                while i < len(dict2[key]):
                    if dict2[key][i] is None:
                        result_dict[key].append(None)
                    else:
                        result_dict[key].append([(dict2[key][i], np.count_nonzero(dict2['labels'] == i))])
                    i+= 1
            elif isinstance(dict2[key], dict):
                assert False, 'merge surfaces and curves not implemented yet'
            else:
                i = 0
                while i < len(result_dict[key]) and i < len(dict2[key]):
                    if result_dict[key][i] is not None and dict2[key][i] is not None:
                        result_dict[key][i].append((dict2[key][i], np.count_nonzero(dict2['labels'] == i)))
                        # verify if is equal or merge parameters 
                    elif dict2[key][i] is not None:
                        result_dict[key][i] = [(dict2[key][i], np.count_nonzero(dict2['labels'] == i))]
                    
                    i+= 1

                while i < len(dict2[key]):
                    if dict2[key][i] is None:
                        result_dict[key].append(None)
                    else:
                        result_dict[key].append([(dict2[key][i], np.count_nonzero(dict2['labels'] == i))])
                    i+= 1
    return result_dict

def mergeByMax(counts, features):
    return features[0]

def mergeByWeightedMean(counts, features):
    total_count = sum(counts)
    weights = [counts[ind]/total_count for ind in range(len(counts))]
    new_f = {}
    for ind, f in enumerate(features):
        for key, value in f.items():
            if isinstance(value, str):
                new_f[key] = value
            elif isinstance(value, bool):
                if key not in new_f:
                    new_f[key] = value
                else:
                    new_f[key] = new_f[key] or value
            elif isinstance(value, list):
                if key not in new_f:
                    new_f[key] = []
                    for k in range(len(value)):
                        new_f[key].append(value[k]*weights[ind])
                else:
                    for k in range(len(value)):
                        new_f[key][k] += value[k]*weights[ind]
            else:
                if key not in new_f:
                    new_f[key] = value*weights[ind]
                else:
                    new_f[key] += value*weights[ind]
    return new_f

def maskValidFeatures(features):
    return np.asarray([not ('invalid' in f and f['invalid'] == True) for f in features])

def mergeFeatures(features, method='max'):
    new_features = [None for _ in range(len(features))]
    for i in range(len(features)):
        if features[i] is not None:
            features_curr = sorted(features[i], key=lambda x: x[1])[::-1]
            just_features = [x[0] for x in features_curr]
            counts = [x[1] for x in features_curr]
            types_count = {}
            types_indices = {}
            for j in range(len(features_curr)):
                tp = just_features[j]['type']
                if tp not in types_count:
                    types_count[tp] = counts[j]
                    types_indices[tp] = [j]
                else:
                    types_count[tp] += counts[j]
                    types_indices[tp].append(j)

            final_tp = None
            final_count = -1
            for tp, count in types_count.items():
                if count > final_count:
                    final_tp = tp
                    final_count = count

            final_indices = types_indices[final_tp]
            just_features = [just_features[ind] for ind in final_indices]
            counts = [counts[ind] for ind in final_indices]

            # special cases, no valid feature or just one valid
            mask_valid = maskValidFeatures(just_features)
            if np.count_nonzero(mask_valid) == 0:
                new_f = just_features[0]
            elif np.count_nonzero(mask_valid) == 1:
                new_f = just_features[np.argmax(mask_valid)]
            else:
                # more than 2 valid features, removing the invalid ones
                just_features = [just_features[ind] for ind in range(len(just_features)) if mask_valid[ind]]
                counts = [counts[ind] for ind in range(len(counts)) if mask_valid[ind]]

                funcs = {
                    'max': mergeByMax,
                    'wm': mergeByWeightedMean
                }

                new_f = funcs[method](counts, just_features)
            
            new_features[i] = new_f

    return new_features

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts a dataset from OBJ and YAML to HDF5')
    parser.add_argument('folder', type=str, help='dataset folder.')
    formats_txt = ','.join(DatasetReaderFactory.READERS_DICT.keys())
    parser.add_argument('input_format', type=str, help=f'types of h5 format to generate. Possible formats: {formats_txt}. Multiple formats can me generated.')
    formats_txt = ','.join(DatasetWriterFactory.WRITERS_DICT.keys())
    parser.add_argument('output_formats', type=str, help='')

    parser.add_argument('--input_gt_format', type=str, help='format of gt data.')

    parser.add_argument('-ct', '--curve_types', type=str, default = '', help='types of curves to generate. Default = ')
    parser.add_argument('-st', '--surface_types', type=str, default = 'plane,cylinder,cone,sphere', help='types of surfaces to generate. Default = plane,cylinder,cone,sphere')
    parser.add_argument('-c', '--centralize', action='store_true', help='')
    parser.add_argument('-a', '--align', action='store_true', help='')
    parser.add_argument('-pnl', '--points_noise_limit', type=float, default = 0., help='')
    parser.add_argument('-nnl', '--normals_noise_limit', type=float, default = 0., help='')
    parser.add_argument('-crf', '--cube_reescale_factor', type=float, default = 0, help='')
    parser.add_argument('-no', '--normalization_order', type=str, default = 'r,c,a,pn,nn,cr', help='')
    parser.add_argument('--use_noisy_points', action='store_true')
    parser.add_argument('--use_noisy_normals', action='store_true')

    for format in DatasetWriterFactory.WRITERS_DICT.keys():
        parser.add_argument(f'-{format}_ct', f'--{format}_curve_types', type=str, help='types of curves to generate. Default = ')
        parser.add_argument(f'-{format}_st', f'--{format}_surface_types', type=str, help='types of surfaces to generate. Default = plane,cylinder,cone,sphere')
        parser.add_argument(f'-{format}_c', f'--{format}_centralize', action='store_true', help='')
        parser.add_argument(f'-{format}_a', f'--{format}_align', action='store_true', help='')
        parser.add_argument(f'-{format}_pnl', f'--{format}_points_noise_limit', type=float, help='')
        parser.add_argument(f'-{format}_nnl', f'--{format}_normals_noise_limit', type=float, help='')
        parser.add_argument(f'-{format}_crf', f'--{format}_cube_reescale_factor', type=float, help='')
        parser.add_argument(f'-{format}_no', f'--{format}_normalization_order', type=str, help='')

    parser.add_argument('--input_dataset_folder_name', type=str, default = 'dataset_divided', help='input dataset folder name.')
    parser.add_argument('--input_data_folder_name', type=str, default = 'data', help='input data folder name.')
    parser.add_argument('--input_gt_dataset_folder_name', type=str, help='input dataset folder name.')
    parser.add_argument('--input_gt_data_folder_name', type=str, help='input gt data folder name.')
    parser.add_argument('--output_dataset_folder_name', type=str, default = 'dataset_merged', help='output dataset folder name.')
    parser.add_argument('--output_data_folder_name', type=str, default = '', help='output data folder name.')
    parser.add_argument('--transform_folder_name', type=str, default = 'transform', help='transform folder name.')
    parser.add_argument('--merge_method', choices=['max', 'wm'], type=str, default = 'wm', help='')

    parser.add_argument('--use_input_gt_transform', action='store_true', help='flag to use transforms from ground truth dataset (not needed if the dataset folder is the same)')
    parser.add_argument('--no_use_output_gt_transform', action='store_false', help='')

    parser.add_argument('--no_use_data_primitives', action='store_true')

    args = vars(parser.parse_args())

    folder_name = args['folder']
    input_format = args['input_format']
    input_gt_format = args['input_gt_format']
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
    input_gt_dataset_folder_name = args['input_gt_dataset_folder_name']
    output_dataset_folder_name = args['output_dataset_folder_name']
    input_data_folder_name = args['input_data_folder_name']
    output_data_folder_name = args['output_data_folder_name']
    output_data_folder_name = input_data_folder_name if output_data_folder_name == '' else output_data_folder_name
    input_gt_data_folder_name = args['input_gt_data_folder_name']
    transform_folder_name = args['transform_folder_name']
    merge_method = args['merge_method']

    use_input_gt_transform = args['use_input_gt_transform']

    use_data_primitives = not args['no_use_data_primitives']
    use_noisy_points = args['use_noisy_points']
    use_noisy_normals = args['use_noisy_normals']

    if input_gt_dataset_folder_name is not None and input_gt_data_folder_name is None:
        input_gt_data_folder_name = input_data_folder_name
    
    if input_gt_data_folder_name is not None and input_gt_dataset_folder_name is None:
        input_gt_dataset_folder_name = input_dataset_folder_name

    if input_gt_format is None:
        input_gt_format = input_format

    input_parameters = {}
    input_gt_parameters = {}
    output_parameters = {}

    assert input_format in DatasetReaderFactory.READERS_DICT.keys()

    input_parameters[input_format] = {}

    input_dataset_format_folder_name = join(folder_name, input_dataset_folder_name, input_format)
    input_parameters[input_format]['dataset_folder_name'] = input_dataset_format_folder_name
    input_data_format_folder_name = join(input_dataset_format_folder_name, input_data_folder_name)
    input_parameters[input_format]['data_folder_name'] = input_data_format_folder_name
    input_transform_format_folder_name = join(input_dataset_format_folder_name, transform_folder_name)
    input_parameters[input_format]['transform_folder_name'] = input_transform_format_folder_name
    input_parameters[input_format]['use_data_primitives'] = use_data_primitives
    input_parameters[input_format]['unnormalize'] = True

    input_gt_transform_format_folder_name = None
    if input_gt_dataset_folder_name is not None and input_gt_data_folder_name is not None:
        input_gt_parameters[input_gt_format] = {}

        input_gt_dataset_format_folder_name = join(folder_name, input_gt_dataset_folder_name, input_gt_format)
        input_gt_parameters[input_gt_format]['dataset_folder_name'] = input_gt_dataset_format_folder_name
        input_gt_data_format_folder_name = join(input_gt_dataset_format_folder_name, input_gt_data_folder_name)
        input_gt_parameters[input_gt_format]['data_folder_name'] = input_gt_data_format_folder_name
        input_gt_transform_format_folder_name = join(input_gt_dataset_format_folder_name, transform_folder_name)
        input_gt_parameters[input_gt_format]['transform_folder_name'] = input_gt_transform_format_folder_name
        input_gt_parameters[input_gt_format]['unnormalize'] = True

    if use_input_gt_transform and input_gt_transform_format_folder_name is not None:
        input_parameters[input_format]['transform_folder_name'] = input_gt_transform_format_folder_name

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
        output_data_format_folder_name = join(output_dataset_format_folder_name, output_data_folder_name)
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
        gt_reader = gt_reader_factory.getReaderByFormat(input_gt_format)
        gt_reader.setCurrentSetName('val')
        query_files = reader.filenames_by_set['val']
        gt_files = gt_reader.filenames_by_set['val']
        assert sorted(query_files) == sorted(gt_files), 'gt has different files from query'
    else:
        gt_reader = None

    dataset_writer_factory = DatasetWriterFactory(output_parameters)
    dataset_writer_factory.setCurrentSetNameAllFormats('val')     

    files_dict = getMergedFilesDict(reader.filenames_by_set['val'])
    
    fs = ['uploads_files_98611_3D_offshore_oil_tanker_dock'] #['27','3D-In Lined Calciner (ILC)-Steel Building','76.Skid_XL-60','Assem1','Assem1  with accurate Skid','Chiller NH3 for brine_03','Condensate_Module','russ','uploads_files_98369_mooring_dock_with_bridge','uploads_files_98408_fuel_gas_scrubber','uploads_files_98448_contango_111106c-3d_steel','uploads_files_98485_lean_to_jacket','uploads_files_98589_3d_salvage_jacket','uploads_files_98609_firewater_tower_3d','uploads_files_98611_3D_offshore_oil_tanker_dock']

    for f in fs:
        del files_dict[f]

    for merged_filename, divided_filenames in tqdm(files_dict.items(), desc='Generating Merged Models', position=0):
        input_data = {}
        reader.filenames_by_set['val'] = sorted(divided_filenames)
        if gt_reader is not None:
            gt_reader.filenames_by_set['val'] = sorted(divided_filenames)
        global_min = -1
        num_points = 0
        gt_labels = None
        for div_filename in tqdm(divided_filenames, desc=f'Model {merged_filename}', position=1, leave=False):
            data = reader.step()
            data['points'] = data['points'] if not use_noisy_points else data['noisy_points']
            data['normals'] = data['normals'] if not use_noisy_normals else data['noisy_normals']
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

        input_data['features_data'] = mergeFeatures(input_data['features_data'], merge_method)

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
