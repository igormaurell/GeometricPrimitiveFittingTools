import argparse
import json
from os.path import join
from os import makedirs
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from lib.readers import DatasetReaderFactory
from lib.utils import computeFeaturesPointIndices, writeColorPointCloudOBJ, getAllColorsArray, computeRGB
from lib.matching import mergeQueryAndGTData
from lib.evaluator import computeIoUs

from asGeometryOCCWrapper.surfaces import SurfaceFactory

from tqdm.contrib.concurrent import process_map, thread_map

from copy import deepcopy

def printAndReturn(text):
    print(text)
    return text

def generateErrorsBoxPlot(errors, individual=True, all_models=False):
    data_distances = []
    data_angles = []
    data_labels = []
    if all_models:
        pass
    else:
        for tp, e in errors.items():
            data_labels.append(tp)
            data_distances.append(np.concatenate(e['distances']) if len(e['distances']) > 0 else [])
            data_angles.append(np.concatenate(e['angles']) if len(e['angles']) > 0 else [])
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.tight_layout(pad=2.0)
    ax1.set_title('Distance Deviation (m)')
    if len(data_distances) > 0:
        ax1.boxplot(data_distances, labels=data_labels, autorange=False, meanline=True)
    ax2.set_title('Normal Deviation (°)')
    if len(data_angles) > 0:
        ax2.boxplot(data_angles, labels=data_labels, autorange=False, meanline=True)
    return fig

def sumToLogsDict(keys, d, nop=0, novp=0, noip=0, nopoints=0, sd=0, sa=0, siiou=0.):
    for key in keys:
        d[key]['number_of_primitives'] += nop
        d[key]['number_of_void_primitives'] += novp
        d[key]['number_of_invalid_primitives'] += noip
        d[key]['number_of_points'] += nopoints
        d[key]['mean_distance_error'] += sd
        d[key]['mean_normal_error'] += sa
        d[key]['mean_iou'] += siiou
    return d

def getBaseKeyLogsDict():
    d = {
        'number_of_primitives': 0,
        'number_of_void_primitives': 0,
        'number_of_invalid_primitives': 0,
        'number_of_points': 0,
        'mean_distance_error': 0,
        'mean_normal_error': 0,
        'mean_iou': 0,
        }
    return d

def addTwoLogsDict(first, second):
    for key in second:
        if key not in first:
            first[key] = getBaseKeyLogsDict()
        first = sumToLogsDict([key], first, nop=second[key]['number_of_primitives'], novp=second[key]['number_of_void_primitives'],
                              noip=second[key]['number_of_invalid_primitives'], nopoints=second[key]['number_of_points'], 
                              sd=second[key]['mean_distance_error'], sa=second[key]['mean_normal_error'], siiou=second[key]['mean_iou'])
    return first

def generateErrorsLogDict(errors):
    logs_dict = {
        'Total': getBaseKeyLogsDict(),
    }
    for tp, e in errors.items():
        number_of_primitives = len(e['distances']) + len(e['invalid_primitives'])
        number_of_void_primitives = len(e['void_primitives'])
        number_of_invalid_primitives = len(e['invalid_primitives'])
        ind_distances = e['distances']
        ind_angles = e['angles']
        instance_ious = e['instance_ious']

        number_of_points = 0
        summd = 0.
        summa = 0.
        for i in range(len(ind_distances)):
            number_of_points += len(ind_distances[i])
            summd += np.sum(ind_distances[i])
            summa += np.sum(ind_angles[i])
        
        if len(instance_ious) > 0:
            siiou = sum(instance_ious)
        else:
            siiou = -1
        
        if tp not in logs_dict.keys():
            logs_dict[tp] = getBaseKeyLogsDict()
        logs_dict = sumToLogsDict(['Total', tp], logs_dict, nop=number_of_primitives, 
                                  novp=number_of_void_primitives, noip=number_of_invalid_primitives,
                                  nopoints=number_of_points, sd=summd, sa=summa, siiou=siiou)
    return logs_dict

def computeLogMeans(logs_dict, denominator=0):
    result = deepcopy(logs_dict)
    for tp in logs_dict.keys():
        number_points = result[tp]['number_of_points'] if result[tp]['number_of_points'] > 0 else 1
        number_points = number_points if denominator == 0 else denominator

        number_valid_primitives = result[tp]['number_of_primitives'] - result[tp]['number_of_void_primitives']
        number_valid_primitives = number_valid_primitives if denominator == 0 else denominator

        if 'mean_distance_error' in result[tp]:
            result[tp]['mean_distance_error'] = result[tp]['mean_distance_error']/number_points
        if 'mean_normal_error' in result[tp]:
            result[tp]['mean_normal_error'] = result[tp]['mean_normal_error']/number_points
        if 'mean_iou' in result[tp]:
            result[tp]['mean_iou'] = result[tp]['mean_iou']/number_valid_primitives if number_valid_primitives > 0 else 0.

    return result

def filterLog(logs_dict):
    result = deepcopy(logs_dict)
    for tp in logs_dict.keys():
        if 'mean_iou' in result[tp]:
            if result[tp]['mean_iou'] < 0:
                del result[tp]['mean_iou']
    
    return result

folder_name = ''
dataset_folder_name = ''
data_folder_name = ''
result_folder_name = ''
transform_folder_name = ''
VERBOSE = False
write_segmentation_gt = False
write_points_error = False
box_plot = False

def process(data_tuple):
    if len(data_tuple) == 1:
        data = data_tuple[0]
        gt_data = None
    else:
        data, gt_data = data_tuple
        data = mergeQueryAndGTData(data, gt_data)

    filename = data['filename'] if 'filename' in data.keys() else str(i)
    points = data['points']
    normals = data['normals']
    labels = data['labels']
    features = data['features']
    if points is None or normals is None or labels is None or features is None:
        print('Invalid Model.')
        return None
    
    dataset_errors[filename] = {}

    colors_instances = np.zeros(shape=points.shape, dtype=np.int64) + np.array([255, 255, 255])
    colors_types = np.zeros(shape=points.shape, dtype=np.int64) + np.array([255, 255, 255])
    
    fpi = computeFeaturesPointIndices(labels, size=len(features))

    instance_ious = []
    if gt_data is not None:
        query_labels = data['labels']
        gt_labels = gt_data['labels'][data['gt_indices']]
        instance_ious = computeIoUs(query_labels, gt_labels, p=(filename == '1_2_3_0'))
        if filename == '1_2_3_0':
            print([x for x in instance_ious if x > 0])
        
    for i, feature in enumerate(features):
        if feature is not None:
            points_curr = points[fpi[i]]
            normals_curr = normals[fpi[i]]
            primitive = None
            try:
                primitive = SurfaceFactory.fromDict(feature)
                tp = primitive.getType()             
            except:
                tp = feature['type']
                
            if tp not in dataset_errors[filename]:
                dataset_errors[filename][tp] = {'distances': [], 'mean_distances': [], 'angles': [], 'mean_angles': [], 'void_primitives': [],
                                                'invalid_primitives': [], 'instance_ious': []}

            if len(fpi[i]) == 0:
                dataset_errors[filename][tp]['void_primitives'].append(i)
                dataset_errors[filename][tp]['invalid_primitives'].append(i)
            elif primitive is None:
                dataset_errors[filename][tp]['invalid_primitives'].append(i)

            if len(fpi[i]) > 0 and primitive is not None:
                distances, angles = primitive.computeErrors(points_curr, normals=normals_curr)
                dataset_errors[filename][tp]['distances'].append(distances)
                dataset_errors[filename][tp]['angles'].append(angles)
                
            if len(fpi[i]) > 0:
                if gt_data is not None:
                    dataset_errors[filename][tp]['instance_ious'].append(instance_ious[i])

                if write_segmentation_gt:
                    colors_instances[fpi[i], :] = computeRGB(colors_full[i%len(colors_full)])
                    if primitive is not None:
                        color = primitive.getColor()
                    else:
                        color = SurfaceFactory.FEATURES_SURFACE_CLASSES[feature['type']].getColor()
                    colors_types[fpi[i], :] = color
                # if write_points_error:
                #     error_dist, error_ang = computeErrorsArrays(fpi[i], distances, angles)
                #     error_both = sortedIndicesIntersection(error_dist, error_ang)
                #     colors_instances[error_dist, :] = np.array([0, 255, 255])
                #     colors_types[error_dist, :] = np.array([0, 255, 255])
                #     colors_instances[error_ang, :] = np.array([0, 0, 0])
                #     colors_types[error_ang, :] = np.array([0, 0, 0])
                #     colors_instances[error_both, :] = np.array([255, 0, 255])
                #     colors_types[error_both, :] = np.array([255, 0, 255])

    logs_dict = generateErrorsLogDict(dataset_errors[filename])
    logs_dict_final = computeLogMeans(logs_dict)
    logs_dict_final['Total']['number_of_points'] += np.count_nonzero(labels==-1)

    with open(f'{log_format_folder_name}/{filename}.json', 'w') as f:
        json.dump(filterLog(logs_dict_final), f, indent=4)
    
    if write_segmentation_gt:
        instances_filename = f'{filename}_instances.obj'
        #points, _, _ = unNormalize(points, transforms, invert=False)
        writeColorPointCloudOBJ(join(seg_format_folder_name, instances_filename), np.concatenate((points, colors_instances), axis=1))
        types_filename = f'{filename}_types.obj'
        writeColorPointCloudOBJ(join(seg_format_folder_name, types_filename), np.concatenate((points, colors_types), axis=1))
    if box_plot:
        fig = generateErrorsBoxPlot(dataset_errors[filename])
        plt.figure(fig.number)
        plt.savefig(f'{box_plot_format_folder_name}/{filename}.png')
    
    return logs_dict_final

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Geometric Primitive Fitting Results, works for dataset validation and for evaluate predictions')
    parser.add_argument('folder', type=str, help='dataset folder.')
    formats_txt = ','.join(DatasetReaderFactory.READERS_DICT.keys())
    parser.add_argument('format', type=str, help=f'types of h5 format to generate. Possible formats: {formats_txt}. Multiple formats can me generated.')

    parser.add_argument('--dataset_folder_name', type=str, default = 'dataset', help='input dataset folder name.')
    parser.add_argument('--gt_dataset_folder_name', type=str, help='gt dataset folder name.')
    parser.add_argument('--data_folder_name', type=str, default = 'data', help='data folder name.')
    parser.add_argument('--result_folder_name', type=str, default = 'eval', help='evaluation folder name.')
    parser.add_argument('--gt_data_folder_name', type=str, help='input gt data folder name.')
    parser.add_argument('--transform_folder_name', type=str, default = 'transform', help='transform folder name.')

    parser.add_argument('-v', '--verbose', action='store_true', help='show more verbose logs.')
    parser.add_argument('-s', '--segmentation_gt', action='store_true', help='write segmentation ground truth.')
    parser.add_argument('-p', '--points_error', action='store_true', help='write segmentation ground truth.')
    parser.add_argument('-b', '--show_box_plot', action='store_true', help='show box plot of the data.')

    parser.add_argument('--use_gt_transform', action='store_true', help='flag to use transforms from ground truth dataset (not needed if the dataset folder is the same)')

    args = vars(parser.parse_args())

    folder_name = args['folder']
    format = args['format']
    dataset_folder_name = args['dataset_folder_name']
    gt_dataset_folder_name = args['gt_dataset_folder_name']
    data_folder_name = args['data_folder_name']
    gt_data_folder_name = args['gt_data_folder_name']
    result_folder_name = args['result_folder_name']
    transform_folder_name = args['transform_folder_name']

    VERBOSE = args['verbose']
    write_segmentation_gt = args['segmentation_gt']
    write_points_error = args['points_error']
    box_plot = args['show_box_plot']

    use_gt_transform = args['use_gt_transform']

    if gt_dataset_folder_name is not None and gt_data_folder_name is None:
        gt_data_folder_name = data_folder_name
    
    if gt_data_folder_name is not None and gt_dataset_folder_name is None:
        gt_dataset_folder_name = dataset_folder_name

    parameters = {}
    gt_parameters = {}

    assert format in DatasetReaderFactory.READERS_DICT.keys()

    parameters[format] = {}
    dataset_format_folder_name = join(folder_name, dataset_folder_name, format)
    parameters[format]['dataset_folder_name'] = dataset_format_folder_name
    data_format_folder_name = join(dataset_format_folder_name, data_folder_name)
    parameters[format]['data_folder_name'] = data_format_folder_name
    transform_format_folder_name = join(dataset_format_folder_name, transform_folder_name)
    parameters[format]['transform_folder_name'] = transform_format_folder_name

    gt_transform_format_folder_name = None
    if gt_dataset_folder_name is not None and gt_data_folder_name is not None:
        gt_parameters[format] = {}

        gt_dataset_format_folder_name = join(folder_name, gt_dataset_folder_name, format)
        gt_parameters[format]['dataset_folder_name'] = gt_dataset_format_folder_name
        gt_data_format_folder_name = join(gt_dataset_format_folder_name, gt_data_folder_name)
        gt_parameters[format]['data_folder_name'] = gt_data_format_folder_name
        gt_transform_format_folder_name = join(gt_dataset_format_folder_name, transform_folder_name)
        gt_parameters[format]['transform_folder_name'] = gt_transform_format_folder_name

    if use_gt_transform and gt_transform_format_folder_name is not None:
        parameters[format]['transform_folder_name'] = gt_transform_format_folder_name

    dataset_reader_factory = DatasetReaderFactory(parameters)
    reader = dataset_reader_factory.getReaderByFormat(format)

    gt_reader = None
    if len(gt_parameters) > 0:
        gt_dataset_reader_factory = DatasetReaderFactory(gt_parameters)
        gt_reader = gt_dataset_reader_factory.getReaderByFormat(format)

    result_format_folder_name = join(dataset_format_folder_name, result_folder_name)
    makedirs(result_format_folder_name, exist_ok=True)

    seg_format_folder_name = join(result_format_folder_name, 'seg')
    if write_segmentation_gt:
        makedirs(seg_format_folder_name, exist_ok=True)

    box_plot_format_folder_name = join(result_format_folder_name, 'boxplot')
    if box_plot:
        makedirs(box_plot_format_folder_name, exist_ok=True)

    log_format_folder_name = join(result_format_folder_name, 'log')
    makedirs(log_format_folder_name, exist_ok=True)

    sets = ['val', 'train']
    colors_full = getAllColorsArray()
    for s in sets:
        reader.setCurrentSetName(s)
        if gt_reader is not None:
            gt_reader.setCurrentSetName(s)
            files = reader.filenames_by_set['val']
            gt_files = gt_reader.filenames_by_set['val']
            assert sorted(files) == sorted(gt_files), f'\n {sorted(files)} \n {sorted(gt_files)}'
            gt_reader.filenames_by_set['val'] = deepcopy(files)
            readers = zip(reader, gt_reader)
        else:
            readers = zip(reader)

        dataset_errors = {}
        full_logs_dicts = {}

        results = process_map(process, readers, max_workers=32, chunksize=1)

        print('Accumulating...')
        c = 0
        for logs_dict in tqdm(results):
            #print(reader.filenames_by_set['val'][c], logs_dict['Total']['mean_iou'])
            full_logs_dicts = addTwoLogsDict(full_logs_dicts, logs_dict)
            #c+= 1

        print(len(results))

        with open(f'{log_format_folder_name}/{s}.json', 'w') as f:
            json.dump(filterLog(computeLogMeans(full_logs_dicts, denominator=len(results))), f, indent=4)