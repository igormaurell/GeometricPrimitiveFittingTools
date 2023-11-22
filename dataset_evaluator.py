import argparse
import json
from os.path import join, exists
from os import makedirs
import numpy as np
from shutil import rmtree
from tqdm import tqdm
import matplotlib.pyplot as plt
from lib.readers import DatasetReaderFactory
from lib.utils import computeFeaturesPointIndices, writeColorPointCloudOBJ, getAllColorsArray, computeRGB
from lib.matching import mergeQueryAndGTData
from lib.evaluator import computeIoUs
from lib.primitives import ResidualLoss
from lib.normalization import rescale, cubeRescale
from pprint import pprint
from math import ceil

from asGeometryOCCWrapper.surfaces import SurfaceFactory

from tqdm.contrib.concurrent import process_map, thread_map

from copy import deepcopy

def printAndReturn(text):
    print(text)
    return text

def generateErrorsBoxPlot(errors, distances_key='distances', angles_key='angles'):
    data_distances = []
    data_angles = []
    data_labels = []
    for tp, e in errors.items():
        data_labels.append(tp)
        data_distances.append(np.concatenate(e[distances_key]) if len(e[distances_key]) > 0 else [])
        data_angles.append(np.concatenate(e[angles_key]) if len(e[angles_key]) > 0 else [])
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.tight_layout(pad=2.0)
    ax1.set_title('Distance Deviation (m)')
    if len(data_distances) > 0:
        ax1.boxplot(data_distances, labels=data_labels, autorange=False, meanline=True)
    ax2.set_title('Normal Deviation (°)')
    if len(data_angles) > 0:
        ax2.boxplot(data_angles, labels=data_labels, autorange=False, meanline=True)
    return fig

def sumToLogsDict(keys, d, nop=0, novp=0, noip=0, nopoints=0, novpoints=0, nogtpoints=0,
                  sd=0, sa=0, sdgt=0, sagt=0, siiou=0., stiou=0.):
    for key in keys:
        d[key]['number_of_primitives'] += nop
        d[key]['number_of_void_primitives'] += novp
        d[key]['number_of_invalid_primitives'] += noip
        d[key]['number_of_points'] += nopoints
        d[key]['number_of_valid_points'] += novpoints
        d[key]['number_of_valid_gt_points'] += nogtpoints
        d[key]['mean_distance_error'] += sd
        d[key]['mean_normal_error'] += sa        
        d[key]['mean_iou'] += siiou
        d[key]['mean_type_iou'] += stiou
        d[key]['mean_distance_gt_error'] += sdgt
        d[key]['mean_normal_gt_error'] += sagt  
    return d

def getBaseKeyLogsDict():
    d = {
        'number_of_primitives': 0,
        'number_of_void_primitives': 0,
        'number_of_invalid_primitives': 0,
        'number_of_points': 0,
        'number_of_valid_points': 0,
        'number_of_valid_gt_points': 0,
        'mean_distance_error': 0,
        'mean_normal_error': 0,
        'mean_iou': 0,
        'mean_type_iou': 0,
        'mean_distance_gt_error': 0,
        'mean_normal_gt_error': 0
        }
    return d

def addTwoLogsDict(first, second):
    for key in second:
        if key not in first:
            first[key] = getBaseKeyLogsDict()
        first = sumToLogsDict([key], first, nop=second[key]['number_of_primitives'], novp=second[key]['number_of_void_primitives'],
                              noip=second[key]['number_of_invalid_primitives'], nopoints=second[key]['number_of_points'], 
                              novpoints=second[key]['number_of_valid_points'], nogtpoints=second[key]['number_of_valid_gt_points'],
                              sd=second[key]['mean_distance_error'], sdgt=second[key]['mean_distance_gt_error'],
                              sagt=second[key]['mean_normal_gt_error'], sa=second[key]['mean_normal_error'], siiou=second[key]['mean_iou'],
                              stiou=second[key]['mean_type_iou'])
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
        type_ious = e['type_ious']

        number_of_points = e['number_of_points']
        number_of_valid_points = 0
        summd = 0.
        summa = 0.
        for i in range(len(ind_distances)):
            number_of_valid_points += len(ind_distances[i])
            summd += np.nanmean(ind_distances[i]) if len(ind_distances[i]) > 0 else 0.
            summa += np.nanmean(ind_angles[i]) if len(ind_angles[i]) > 0 else 0.

        number_of_valid_gt_points = 0
        ind_gt_distances = e['distances_to_gt']
        ind_gt_angles = e['angles_to_gt']
        summdgt = 0.
        summagt = 0.
        if len(ind_gt_distances) > 0:
            summdgt = 0.
            summagt = 0.
            for i in range(len(ind_gt_distances)):
                number_of_valid_gt_points += len(ind_gt_angles[i])
                summdgt += np.nanmean(ind_gt_distances[i]) if len(ind_gt_distances[i]) > 0 else 0.
                summagt += np.nanmean(ind_gt_angles[i]) if len(ind_gt_angles[i]) > 0 else 0.
        else:
            summdgt = -1.
            summagt = -1.
        
        if len(instance_ious) > 0:
            siiou = sum(instance_ious)
        else:
            siiou = -1
        
        if len(type_ious) > 0:
            stiou = sum(type_ious)
        else:
            stiou = -1
        
        if tp not in logs_dict.keys():
            logs_dict[tp] = getBaseKeyLogsDict()
        logs_dict = sumToLogsDict(['Total', tp], logs_dict, nop=number_of_primitives, 
                                  novp=number_of_void_primitives, noip=number_of_invalid_primitives,
                                  nopoints=number_of_points, novpoints=number_of_valid_points,
                                  nogtpoints=number_of_valid_gt_points, sd=summd, sa=summa, sdgt=summdgt, sagt=summagt,
                                  siiou=siiou, stiou=stiou)
    return logs_dict

def computeLogMeans(logs_dict, denominator=0):
    result = deepcopy(logs_dict)
    for tp in logs_dict.keys():
        #number_valid_points = result[tp]['number_of_valid_points'] if result[tp]['number_of_valid_points'] > 0 else 1
        #number_valid_points = number_valid_points if denominator == 0 else denominator

        #number_of_valid_gt_points = result[tp]['number_of_valid_gt_points'] if result[tp]['number_of_valid_gt_points'] > 0 else 1
        #number_of_valid_gt_points = number_of_valid_gt_points if denominator == 0 else denominator

        number_of_primitives = result[tp]['number_of_primitives'] - result[tp]['number_of_void_primitives']
        number_of_primitives = number_of_primitives if denominator == 0 else denominator

        number_of_valid_primitives = result[tp]['number_of_primitives'] - result[tp]['number_of_invalid_primitives']
        number_of_valid_primitives = number_of_valid_primitives if denominator == 0 else denominator

        if 'mean_distance_error' in result[tp]:
            result[tp]['mean_distance_error'] = result[tp]['mean_distance_error']/number_of_valid_primitives if number_of_valid_primitives > 0 else 0.
        if 'mean_normal_error' in result[tp]:
            result[tp]['mean_normal_error'] = result[tp]['mean_normal_error']/number_of_valid_primitives if number_of_valid_primitives > 0 else 0.
        if 'mean_distance_gt_error' in result[tp]:
            result[tp]['mean_distance_gt_error'] = result[tp]['mean_distance_gt_error']/number_of_valid_primitives if number_of_valid_primitives > 0 else 0.
        if 'mean_normal_gt_error' in result[tp]:
            result[tp]['mean_normal_gt_error'] = result[tp]['mean_normal_gt_error']/number_of_valid_primitives if number_of_valid_primitives > 0 else 0.
        if 'mean_iou' in result[tp]:
            result[tp]['mean_iou'] = result[tp]['mean_iou']/number_of_primitives if number_of_primitives > 0 else 0.
        if 'mean_type_iou' in result[tp]:
            result[tp]['mean_type_iou'] = result[tp]['mean_type_iou']/number_of_primitives if number_of_primitives > 0 else 0.

    return result

def filterLog(logs_dict):
    result = deepcopy(logs_dict)
    for tp in logs_dict.keys():
        if 'mean_distance_gt_error' in result[tp]:
            if result[tp]['mean_distance_gt_error'] < 0:
                del result[tp]['mean_distance_gt_error']
        if 'mean_normal_gt_error' in result[tp]:
            if result[tp]['mean_normal_gt_error'] < 0:
                del result[tp]['mean_normal_gt_error']
        if 'mean_normal_gt_error' not in result[tp] and 'mean_distance_gt_error' not in result[tp]:
            del result[tp]['number_of_valid_gt_points']
        if 'mean_iou' in result[tp]:
            if result[tp]['mean_iou'] < 0:
                del result[tp]['mean_iou']
        if 'mean_type_iou' in result[tp]:
            if result[tp]['mean_type_iou'] < 0:
                del result[tp]['mean_type_iou']
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
ignore_primitives_orientation = False

def process(data_tuple):
    if len(data_tuple) == 1:
        data = data_tuple[0]
        gt_data = None
    else:
        data, gt_data = data_tuple
        data = mergeQueryAndGTData(data, gt_data, force_match=force_match)

    residual_distance = ResidualLoss()

    filename = data['filename'] if 'filename' in data.keys() else str(i)
    points = data['noisy_points'] if use_noisy_points else data['points']
    normals = data['noisy_normals'] if use_noisy_normals else data['normals']

    labels = data['labels']
    labels[labels < -1] = -1 # removing non_gt_features

    features = data['features_data']
    if points is None or normals is None or labels is None or features is None:
        print('Invalid Model.')
        return None
    
    dataset_errors = {}

    colors_instances = np.zeros(shape=points.shape, dtype=np.int64) + np.array([255, 255, 255])
    colors_types = np.zeros(shape=points.shape, dtype=np.int64) + np.array([255, 255, 255])
    
    fpi = computeFeaturesPointIndices(labels, size=len(features))

    instance_ious = []
    fpi_gt = None
    gt_normals = None
    #type_ious = []
    if gt_data is not None:
        query_labels = data['labels']
        gt_labels = gt_data['labels'][data['gt_indices']]
        instance_ious = computeIoUs(query_labels, gt_labels)

        fpi_gt = computeFeaturesPointIndices(gt_data['labels'], size=len(gt_data['features_data']))

        gt_points = gt_data['points']
        gt_normals = gt_data['normals']
    
    reescale_factor = 1.
    if cube_reescale_factor > 0:
        if gt_points is not None:
            _, _, reescale_factor = cubeRescale(gt_points.copy())
        else:
            _, _, reescale_factor = cubeRescale(points.copy())
    
    if not no_use_occ_geometries:
        points, features, _ = rescale(points, features=features, factor=1000)
        if gt_data is not None:
            gt_points, _, _ = rescale(gt_points, features=[], factor=1000)  
        reescale_factor /= 1000

    for i, feature in enumerate(features):
        indices = fpi[i]
        if feature is not None and indices is not None:
            points_curr = points[indices]
            normals_curr = normals[indices]
            reescale_factor_curr = reescale_factor
            primitive = None
            if not no_use_occ_geometries:
                try:
                    primitive = SurfaceFactory.fromDict(feature)
                    tp = primitive.getType()
                except:
                    pass
            
            if primitive is None:
                tp = feature['type']
                
            if tp not in dataset_errors:
                dataset_errors[tp] = {'number_of_points': 0, 'distances': [], 'angles': [],
                                      'distances_to_gt': [], 'angles_to_gt': [], 'void_primitives': [],
                                      'invalid_primitives': [], 'instance_ious': [], 'type_ious': []}

            dataset_errors[tp]['number_of_points'] += len(indices)

            if len(indices) == 0:
                dataset_errors[tp]['void_primitives'].append(i)
                dataset_errors[tp]['invalid_primitives'].append(i)
            #elif primitive is None and not no_use_occ_geometries:
            #    dataset_errors[tp]['invalid_primitives'].append(i)
            elif 'invalid' in feature and feature['invalid']:
                dataset_errors[tp]['invalid_primitives'].append(i)

            if len(indices) > 0 and ('invalid' not in feature or not feature['invalid']):
                if not no_use_occ_geometries and primitive is not None:
                    distances, angles = primitive.computeErrors(points_curr, normals=normals_curr,
                                                                symmetric_normals=ignore_primitives_orientation)
                    print(len(distances), np.all(np.isnan(distances)))
                else:
                    if not no_use_occ_geometries:
                        points_curr, features_curr, _ = rescale(points_curr, features=[feature], factor=1/1000)
                        feature = features_curr[0]
                        reescale_factor_curr *= 1000
                    distances = residual_distance.residual_loss(points_curr, feature)
                    angles = []
                
                distances*= reescale_factor_curr

                dataset_errors[tp]['distances'].append(distances)
                dataset_errors[tp]['angles'].append(angles)

                if fpi_gt is not None:
                    indices_gt = fpi_gt[i]
                    points_gt_curr = gt_points[indices_gt]
                    normals_gt_curr = gt_normals[indices_gt]
                    
                    if not no_use_occ_geometries and primitive is not None:
                        distances_to_gt, angles_to_gt = primitive.computeErrors(points_gt_curr, normals=normals_gt_curr,
                                                                                symmetric_normals=ignore_primitives_orientation)
                    else:
                        if not no_use_occ_geometries:
                            points_gt_curr, _, _ = rescale(points_gt_curr, factor=1/1000)
                        distances_to_gt = residual_distance.residual_loss(points_gt_curr, feature)
                        angles_to_gt = []
                
                    distances_to_gt*= reescale_factor_curr

                    dataset_errors[tp]['distances_to_gt'].append(distances_to_gt.astype(np.float32))
                    dataset_errors[tp]['angles_to_gt'].append(angles_to_gt)
                
            if len(indices) > 0:
                if gt_data is not None:
                    dataset_errors[tp]['instance_ious'].append(instance_ious[i])
                    gt_tp = gt_data['features_data'][i]['type']
                    dataset_errors[tp]['type_ious'].append(tp==gt_tp)

                if write_segmentation_gt:
                    colors_instances[indices, :] = computeRGB(colors_full[i%len(colors_full)])
                    if primitive is not None:
                        color = primitive.getColor()
                    else:
                        color = SurfaceFactory.FEATURES_SURFACE_CLASSES[feature['type']].getColor()
                    colors_types[indices, :] = color
                # if write_points_error:
                #     error_dist, error_ang = computeErrorsArrays(indices, distances, angles)
                #     error_both = sortedIndicesIntersection(error_dist, error_ang)
                #     colors_instances[error_dist, :] = np.array([0, 255, 255])
                #     colors_types[error_dist, :] = np.array([0, 255, 255])
                #     colors_instances[error_ang, :] = np.array([0, 0, 0])
                #     colors_types[error_ang, :] = np.array([0, 0, 0])
                #     colors_instances[error_both, :] = np.array([255, 0, 255])
                #     colors_types[error_both, :] = np.array([255, 0, 255])
    
    logs_dict = generateErrorsLogDict(dataset_errors)
    logs_dict_final = computeLogMeans(logs_dict)
    logs_dict_final['Total']['number_of_points'] += np.count_nonzero(labels==-1)

    filtered_logs_dict_final = filterLog(logs_dict_final)

    with open(f'{log_format_folder_name}/{filename}.json', 'w') as f:
        json.dump(filtered_logs_dict_final, f, indent=4)
    
    if write_segmentation_gt:
        instances_filename = f'{filename}_instances.obj'
        #points, _, _ = applyTransforms(points, transforms, invert=False)
        if not no_use_occ_geometries:
            points /= 1000.
        writeColorPointCloudOBJ(join(seg_format_folder_name, instances_filename), np.concatenate((points, colors_instances), axis=1))
        types_filename = f'{filename}_types.obj'
        writeColorPointCloudOBJ(join(seg_format_folder_name, types_filename), np.concatenate((points, colors_types), axis=1))
    if box_plot:
        fig = generateErrorsBoxPlot(dataset_errors)
        plt.figure(fig.number)
        plt.savefig(f'{box_plot_format_folder_name}/{filename}.png')
        plt.close()
        fig2 = generateErrorsBoxPlot(dataset_errors, distances_key='distances_to_gt', angles_key='angles_to_gt')
        plt.figure(fig2.number)
        plt.savefig(f'{box_plot_format_folder_name}/{filename}_to_gt.png')
        plt.close()
    
    return logs_dict_final

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Geometric Primitive Fitting Results, works for dataset validation and for evaluate predictions')
    parser.add_argument('folder', type=str, help='dataset folder.')
    formats_txt = ','.join(DatasetReaderFactory.READERS_DICT.keys())
    parser.add_argument('format', type=str, help=f'types of h5 format to generate. Possible formats: {formats_txt}. Multiple formats can me generated.')

    parser.add_argument('--gt_format', type=str, help='format of gt data.')

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
    parser.add_argument('--no_use_data_primitives', action='store_true')
    parser.add_argument('--force_match', action='store_true')
    parser.add_argument('--use_noisy_points', action='store_true')
    parser.add_argument('--use_noisy_normals', action='store_true')
    parser.add_argument('--no_use_occ_geometries', action='store_true')
    parser.add_argument('--ignore_primitives_orientation', action='store_true')
    parser.add_argument('-w', '--workers', type=int, default=20, help='')
    parser.add_argument('-un', '--unnormalize', action='store_true', help='')
    parser.add_argument('-crf', '--cube_reescale_factor', type=float, default = 0, help='')

    args = vars(parser.parse_args())

    folder_name = args['folder']
    input_format = args['format']
    gt_format = args['gt_format']
    #gt_format = format if gt_format is None else gt_format
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
    use_data_primitives = not args['no_use_data_primitives']
    use_noisy_points = args['use_noisy_points']
    use_noisy_normals = args['use_noisy_normals']
    no_use_occ_geometries = args['no_use_occ_geometries']
    ignore_primitives_orientation = args['ignore_primitives_orientation']
    workers = args['workers']
    unnormalize = args['unnormalize']
    cube_reescale_factor = args['cube_reescale_factor']
    force_match = args['force_match']

    if gt_dataset_folder_name is not None or gt_data_folder_name is not None or gt_format is not None:
        if gt_data_folder_name is None:
            gt_data_folder_name = data_folder_name
        if gt_dataset_folder_name is None:
            gt_dataset_folder_name = dataset_folder_name
        if gt_format is None:
            gt_format = input_format

    parameters = {}
    gt_parameters = {}

    assert input_format in DatasetReaderFactory.READERS_DICT.keys()

    parameters[input_format] = {}
    dataset_format_folder_name = join(folder_name, dataset_folder_name, input_format)
    parameters[input_format]['dataset_folder_name'] = dataset_format_folder_name
    data_format_folder_name = join(dataset_format_folder_name, data_folder_name)
    parameters[input_format]['data_folder_name'] = data_format_folder_name
    transform_format_folder_name = join(dataset_format_folder_name, transform_folder_name)
    parameters[input_format]['transform_folder_name'] = transform_format_folder_name
    parameters[input_format]['use_data_primitives'] = use_data_primitives
    parameters[input_format]['unnormalize'] = unnormalize

    gt_transform_format_folder_name = None
    if gt_dataset_folder_name is not None and gt_data_folder_name is not None:
        gt_parameters[gt_format] = {}

        gt_dataset_format_folder_name = join(folder_name, gt_dataset_folder_name, gt_format)
        gt_parameters[gt_format]['dataset_folder_name'] = gt_dataset_format_folder_name
        gt_data_format_folder_name = join(gt_dataset_format_folder_name, gt_data_folder_name)
        gt_parameters[gt_format]['data_folder_name'] = gt_data_format_folder_name
        gt_transform_format_folder_name = join(gt_dataset_format_folder_name, transform_folder_name)
        gt_parameters[gt_format]['transform_folder_name'] = gt_transform_format_folder_name
        gt_parameters[gt_format]['unnormalize'] = unnormalize

    if use_gt_transform and gt_transform_format_folder_name is not None:
        parameters[gt_format]['transform_folder_name'] = gt_transform_format_folder_name

    dataset_reader_factory = DatasetReaderFactory(parameters)
    reader = dataset_reader_factory.getReaderByFormat(input_format)

    gt_reader = None
    if len(gt_parameters) > 0:
        gt_dataset_reader_factory = DatasetReaderFactory(gt_parameters)
        gt_reader = gt_dataset_reader_factory.getReaderByFormat(gt_format)

    result_format_folder_name = join(dataset_format_folder_name, result_folder_name)
    if exists(result_format_folder_name):
        rmtree(result_format_folder_name)

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
        size = len(reader.filenames_by_set[s])
        reader.filenames_by_set[s] = reader.filenames_by_set[s]
        if size == 0:
            continue
        if gt_reader is not None:
            gt_reader.setCurrentSetName(s)
            files = reader.filenames_by_set[s]
            gt_files = gt_reader.filenames_by_set[s]
            if sorted(files) != sorted(gt_files):
                print(f'Pred has {len(sorted(files))} files and GT has {len(sorted(gt_files))} files.')
                continue
            gt_reader.filenames_by_set[s] = deepcopy(files)
            readers = zip(reader, gt_reader)
        else:
            readers = zip(reader)

        full_logs_dicts = {}

        max_workers = min(size, workers)
        chunksize = ceil(size/max_workers)

        results = process_map(process, readers, max_workers=max_workers, chunksize=chunksize)
        #results = [process(data) for data in tqdm(readers)]

        print('Accumulating...')
        c = 0
        dataset_error_dict = {}
        for logs_dict in tqdm(results):
            for k, v in logs_dict.items():
                if k not in dataset_error_dict:
                    dataset_error_dict[k] = {}
                    for k2, v2 in v.items():
                        dataset_error_dict[k][k2] = [[v2]]
                else:
                    for k2, v2 in v.items():
                        dataset_error_dict[k][k2].append([v2])

            #print(reader.filenames_by_set['val'][c], logs_dict['Total']['mean_iou'])
            full_logs_dicts = addTwoLogsDict(full_logs_dicts, logs_dict)
            #c+= 1

        if box_plot:
            fig = generateErrorsBoxPlot(dataset_error_dict, distances_key='mean_distance_error', angles_key='mean_normal_error')
            plt.figure(fig.number)
            plt.savefig(f'{box_plot_format_folder_name}/{s}.png')
            plt.close()
            fig2 = generateErrorsBoxPlot(dataset_error_dict, distances_key='mean_distance_gt_error', angles_key='mean_normal_gt_error')
            plt.figure(fig2.number)
            plt.savefig(f'{box_plot_format_folder_name}/{s}_to_gt.png')
            plt.close()
        
        final_json = filterLog(computeLogMeans(full_logs_dicts, denominator=len(results)))

        with open(f'{log_format_folder_name}/{s}.json', 'w') as f:
            json.dump(final_json, f, indent=4)

        pprint(final_json)