import argparse
from os.path import join
from os import makedirs
from shutil import rmtree
from math import pi
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from lib.readers import DatasetReaderFactory
from lib.utils import sortedIndicesIntersection, computeFeaturesPointIndices, writeColorPointCloudOBJ, getAllColorsArray, computeRGB
from lib.normalization import unNormalize

from asGeometryOCCWrapper.surfaces import SurfaceFactory

from tqdm.contrib.concurrent import process_map, thread_map
from functools import partial

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

def sumToLogsDict(keys, d, nop=0, novp=0, nopoints=0, ned=0, nea=0, sd=0, sa=0):
    for key in keys:
        d[key]['number_of_primitives'] += nop
        d[key]['number_of_void_primitives'] += novp
        d[key]['number_of_points'] += nopoints
        d[key]['number_error_distances'] += ned
        d[key]['number_error_angles'] += nea
        d[key]['sum_distances'] += sd
        d[key]['sum_angles'] += sa
    return d

def getBaseKeyLogsDict():
    d = {
        'number_of_primitives': 0,
        'number_of_void_primitives': 0,
        'number_of_points': 0,
        'number_error_distances': 0,
        'number_error_angles': 0,
        'sum_distances': 0,
        'sum_angles': 0,
        }
    return d

def addTwoLogsDict(first, second):
    for key in second:
        if key not in first:
            first[key] = getBaseKeyLogsDict()
        first = sumToLogsDict([key], first, nop=second[key]['number_of_primitives'], novp=second[key]['number_of_void_primitives'], nopoints=second[key]['number_of_points'], ned=second[key]['number_error_distances'],
                                   nea=second[key]['number_error_angles'], sd=second[key]['sum_distances'], sa=second[key]['sum_angles'])
    return first

def generateErrorsLogDict(errors, max_distance_deviation, max_angle_deviation):
    logs_dict = {
        'total': getBaseKeyLogsDict(),
    }
    for tp, e in errors.items():
        number_of_primitives = len(e['distances'])     
        number_of_void_primitives = len(e['void_primitives'])

        ind_distances = e['distances']
        ind_angles = e['angles']
        number_of_points = 0
        number_error_distances = 0
        number_error_angles = 0
        summd = 0.
        summa = 0.
        for i in range(len(ind_distances)):
            number_of_points += len(ind_distances[i])
            number_error_distances += np.count_nonzero(ind_distances[i] > max_distance_deviation)
            number_error_angles += np.count_nonzero(ind_angles[i] > max_angle_deviation)
            summd += np.sum(ind_distances[i])
            summa += np.sum(ind_angles[i])
        
        if tp not in logs_dict.keys():
            logs_dict[tp] = getBaseKeyLogsDict()
        logs_dict = sumToLogsDict(['total', tp], logs_dict, nop=number_of_primitives, novp=number_of_void_primitives, nopoints=number_of_points, ned=number_error_distances, nea=number_error_angles, sd=summd, sa=summa)
    return logs_dict

def generateLog(logs_dict, max_distance_deviation, max_angle_deviation):
    log = ''
    for key, value in logs_dict.items():
        number_of_primitives = value['number_of_primitives']
        number_of_void_primitives = value['number_of_void_primitives']
        number_of_points = value['number_of_points']
        number_error_distances = value['number_error_distances']
        number_error_angles = value['number_error_angles']
        sum_distances = value['sum_distances']
        sum_angles = value['sum_angles']
        log += f'\n\t- {key}:\n'
        log += f'\t\t- Number of Primitives: {number_of_primitives + number_of_void_primitives}\n'
        log += f'\t\t- Number of Void Primitives: {number_of_void_primitives}\n'
        log += f'\t\t- Number of Points: {number_of_points}\n'
        log += f'\t\t- Distance Error Rate (>{max_distance_deviation}): {((100.0*number_error_distances)/number_of_points) if number_of_points > 0 else 0 }%\n'
        log += f'\t\t- Normal Error Rate (>{max_angle_deviation}): {((100.0*number_error_angles)/number_of_points) if number_of_points > 0 else 0}%\n'
        log += f'\t\t- Mean Distance Error: {(sum_distances/number_of_points) if number_of_points > 0 else 0}\n'
        log += f'\t\t- Mean Normal Error: {(sum_angles/number_of_points) if number_of_points > 0 else 0}\n\n'
    return log

def computeErrorsArrays(indices, distances, angles, max_distance_deviation, max_angle_deviation):
    error_dist = np.sort(indices[distances > max_distance_deviation])
    error_ang = np.sort(indices[angles > max_angle_deviation])
    return error_dist, error_ang

folder_name = ''
dataset_folder_name = ''
data_folder_name = ''
result_folder_name = ''
transform_folder_name = ''
delete_old_data = False
VERBOSE = False
write_segmentation_gt = False
write_points_error = False
box_plot = False
max_distance_deviation = False
max_angle_deviation = False

def process(data):
    log = ''
            
    filename = data['filename'] if 'filename' in data.keys() else str(i)
    points = data['points']
    normals = data['normals']
    labels = data['labels']
    features = data['features']
    if points is None or normals is None or labels is None or features is None:
        print('Invalid Model.')
        return None
    dataset_errors[filename] = {}
    log += f'\n-- File {filename}:\n'
    colors_instances = np.zeros(shape=points.shape, dtype=np.int64) + np.array([255, 255, 255])
    colors_types = np.zeros(shape=points.shape, dtype=np.int64) + np.array([255, 255, 255])
    fpi = computeFeaturesPointIndices(labels, size=len(features))
    impossible_primitives = {}
    for i, feature in enumerate(features):
        if feature is not None:
            points_curr = points[fpi[i]]
            normals_curr = normals[fpi[i]]
            try:
                primitive = SurfaceFactory.fromDict(feature)
            except:
                if feature['type'] not in impossible_primitives:
                    impossible_primitives[feature['type']] = 1
                else:
                    impossible_primitives[feature['type']] += 1
                continue
            tp = primitive.getType()             
            if tp not in dataset_errors[filename]:
                dataset_errors[filename][tp] = {'distances': [], 'mean_distances': [], 'angles': [], 'mean_angles': [], 'void_primitives': []}
            if len(fpi[i]) == 0:
                dataset_errors[filename][tp]['void_primitives'].append(i)
            else:
                distances, angles = primitive.computeErrors(points_curr, normals=normals_curr)
                dataset_errors[filename][tp]['distances'].append(distances)
                dataset_errors[filename][tp]['angles'].append(angles)
                if write_segmentation_gt:
                    colors_instances[fpi[i], :] = computeRGB(colors_full[i%len(colors_full)])
                    colors_types[fpi[i], :] = primitive.getColor()
                    # if write_points_error:
                    #     error_dist, error_ang = computeErrorsArrays(fpi[i], distances, angles, max_distance_deviation, max_angle_deviation)
                    #     error_both = sortedIndicesIntersection(error_dist, error_ang)
                    #     colors_instances[error_dist, :] = np.array([0, 255, 255])
                    #     colors_types[error_dist, :] = np.array([0, 255, 255])
                    #     colors_instances[error_ang, :] = np.array([0, 0, 0])
                    #     colors_types[error_ang, :] = np.array([0, 0, 0])
                    #     colors_instances[error_both, :] = np.array([255, 0, 255])
                    #     colors_types[error_both, :] = np.array([255, 0, 255])
    if len(impossible_primitives) > 0:
        print(f'{filename} impossible primitives: {impossible_primitives}')
    logs_dict = generateErrorsLogDict(dataset_errors[filename], max_distance_deviation, max_angle_deviation)
    logs_dict['total']['number_of_points'] += np.count_nonzero(labels==-1)
    error_log = generateLog(logs_dict, max_distance_deviation, max_angle_deviation)
    log += error_log
    with open(f'{log_format_folder_name}/{filename}.txt', 'w') as f:
        f.write(log)
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
    
    return logs_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Geometric Primitive Fitting Results, works for dataset validation and for methods results')
    parser.add_argument('folder', type=str, help='dataset folder.')
    formats_txt = ','.join(DatasetReaderFactory.READERS_DICT.keys())
    parser.add_argument('format', type=str, help=f'types of h5 format to generate. Possible formats: {formats_txt}. Multiple formats can me generated.')    
    parser.add_argument('--dataset_folder_name', type=str, default = 'dataset', help='input dataset folder name.')
    parser.add_argument('--data_folder_name', type=str, default = 'data', help='data folder name.')
    parser.add_argument('--result_folder_name', type=str, default = 'val', help='validation folder name.')
    parser.add_argument('--transform_folder_name', type=str, default = 'transform', help='transform folder name.')
    parser.add_argument('-d_dt', '--delete_old_data', action='store_true', help='')
    parser.add_argument('-v', '--verbose', action='store_true', help='show more verbose logs.')
    parser.add_argument('-s', '--segmentation_gt', action='store_true', help='write segmentation ground truth.')
    parser.add_argument('-p', '--points_error', action='store_true', help='write segmentation ground truth.')
    parser.add_argument('-b', '--show_box_plot', action='store_true', help='show box plot of the data.')
    parser.add_argument('-md', '--max_distance_deviation', type=float, default=0.05, help='max distance deviation.')
    parser.add_argument('-mn', '--max_angle_deviation', type=float, default=10, help='max normal angle deviation in degrees.')

    args = vars(parser.parse_args())

    folder_name = args['folder']
    format = args['format']
    dataset_folder_name = args['dataset_folder_name']
    data_folder_name = args['data_folder_name']
    result_folder_name = args['result_folder_name']
    transform_folder_name = args['transform_folder_name']
    delete_old_data = args['delete_old_data']
    VERBOSE = args['verbose']
    write_segmentation_gt = args['segmentation_gt']
    write_points_error = args['points_error']
    box_plot = args['show_box_plot']
    max_distance_deviation = args['max_distance_deviation']
    max_angle_deviation = args['max_angle_deviation']

    parameters = {format: {}}
    dataset_format_folder_name = join(folder_name, dataset_folder_name, format)
    parameters[format]['dataset_folder_name'] = dataset_format_folder_name
    data_format_folder_name = join(dataset_format_folder_name, data_folder_name)
    parameters[format]['data_folder_name'] = data_format_folder_name
    transform_format_folder_name = join(dataset_format_folder_name, transform_folder_name)
    parameters[format]['transform_folder_name'] = transform_format_folder_name

    result_format_folder_name = join(dataset_format_folder_name, result_folder_name)
    if delete_old_data:
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

    dataset_reader_factory = DatasetReaderFactory(parameters)
    reader = dataset_reader_factory.getReaderByFormat(format)
    sets = ['val', 'train']
    colors_full = getAllColorsArray()
    for s in sets:
        reader.setCurrentSetName(s)
        full_log = printAndReturn(f'\nValidation of {s} dataset:\n')
        dataset_errors = {}
        full_logs_dicts = {}

        results = process_map(process, reader, max_workers=32, chunksize=1)

        print('Accumulating...')
        for logs_dict in tqdm(results):
            full_logs_dicts = addTwoLogsDict(full_logs_dicts, logs_dict)

        full_log += generateLog(full_logs_dicts, max_distance_deviation, max_angle_deviation)
        with open(f'{log_format_folder_name}/{s}.txt', 'w') as f:
            f.write(full_log)