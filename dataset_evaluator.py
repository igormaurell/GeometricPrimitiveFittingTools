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
from copy import deepcopy
from pprint import pprint

from asGeometryOCCWrapper.surfaces import SurfaceFactory

from tqdm.contrib.concurrent import process_map, thread_map

from copy import deepcopy

np.warnings.filterwarnings('ignore')

def printAndReturn(text):
    print(text)
    return text

def generateErrorsBoxPlot(errors, distances_key='distance', angles_key='angle'):
    data_distances = []
    data_angles = []
    data_labels = []
    for tp, e in errors.items():
        data_labels.append(tp)
        data_distances.append(e[distances_key])
        data_angles.append(e[angles_key])
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.tight_layout(pad=2.0)
    ax1.set_title('Distance Deviation (m)')
    if len(data_distances) > 0:
        ax1.boxplot(data_distances, labels=data_labels, autorange=False, meanline=True)
    ax2.set_title('Normal Deviation (°)')
    if len(data_angles) > 0:
        ax2.boxplot(data_angles, labels=data_labels, autorange=False, meanline=True)
    return fig

# TODO: delete this and pass by parameter
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

# TODO: transform Metrics into a class
# The intention here is to make easier to add new metrics
METRICS_DICT = {
    'n_prim_points': {'derivations': {'total': np.nansum, 'mean': np.nanmean}, 'reduction_key': 'total'},
    'n_no_prim_points': {'derivations': {'total': np.nansum, 'mean': np.nanmean}, 'reduction_key': 'total'},
    'n_prim': {'derivations': {'total': np.nansum, 'mean': np.nanmean}, 'reduction_key': 'total'},
    'n_invalid_prim': {'derivations': {'total': np.nansum, 'mean': np.nanmean}, 'reduction_key': 'total'},
    'n_void_prim': {'derivations': {'total': np.nansum, 'mean': np.nanmean}, 'reduction_key': 'total'},
    'distance': {'derivations': {'mean': np.nanmean, 'count': len}, 'reduction_key': 'mean'},
    'angle': {'derivations': {'mean': np.nanmean, 'count': len}, 'reduction_key': 'mean'},
    'gt_distance': {'derivations': {'mean': np.nanmean, 'count': len}, 'need_gt': True, 'reduction_key': 'mean'},
    'gt_angle': {'derivations': {'mean': np.nanmean, 'count': len}, 'need_gt': True, 'reduction_key': 'mean'},
    'instance_iou': {'derivations': {'mean': np.nanmean}, 'need_gt': True, 'reduction_key': 'mean'},
    'type_iou': {'derivations': {'mean': np.nanmean}, 'need_gt': True, 'reduction_key': 'mean'}, 
}

# Creating a base metrics dict (with void lists)
def get_base_metrics_dict(with_gt=True):
    d_list = []
    for key in METRICS_DICT:
        need_gt = 'need_gt' in METRICS_DICT[key] and METRICS_DICT[key]['need_gt']
        if not need_gt or (need_gt and with_gt):
            d_list.append((key, []))
    return dict(d_list)

def metrics_dict_list2array(d):
    new_d = {}
    for tp, d2 in d.items():
        new_d[tp] = {}
        for key, value in d2.items():
            new_d[tp][key] = np.asarray(value)
    return new_d

def generate_total_key_metrics_dict(d):
    total_dict = {}
    for key, value in d.items():
        for key2, value2 in value.items():
            if key2 not in total_dict:
                total_dict[key2] = []
            total_dict[key2] += value2
    d['Total'] = total_dict
    return d

def compute_derived_metrics(d):
    derived_metrics_dict = {}
    for tp, d2 in d.items():
        derived_metrics_dict[tp] = {}
        for key, value in METRICS_DICT.items():
            if key in d2:
                for name, func in value['derivations'].items():
                    derived_metrics_dict[tp][f'{name}_{key}'] = func(d2[key])
    return derived_metrics_dict

def reduce_derived_model_metrics(d):
    reduction_maps = dict([(f"{value['reduction_key']}_{key}", key) for key, value in METRICS_DICT.items()])
    reduced_mm = {}
    for tp, d2 in d.items():
        reduced_mm[tp] = {}
        for key, value in d2.items():
            if key in reduction_maps:
                reduced_mm[tp][reduction_maps[key]] = [value]
    return reduced_mm

def concatenate_metrics_dict(dicts):
    final_dict = {}
    for d in dicts:
        for key, value in d.items():
            if key not in final_dict:
                final_dict[key] = {}
            for key2, value2 in value.items():
                if key2 not in final_dict[key]:
                    final_dict[key][key2] = []
                final_dict[key][key2] += value2
    return final_dict

def compute_deviations(points, normals, feature, reescale_factor=1):
    distances = np.empty((points.shape[0],))
    distances[:] = np.nan
    angles = np.empty((normals.shape[0],))
    angles[:] = np.nan

    reescale_factor_curr = 1

    if use_occ:
        points, features_curr, _ = rescale(points, features=[feature], factor=1000)
        feature = features_curr[0]
        reescale_factor_curr *= 1000

        try:
            ## TODO: fix add copy inside asGeometryOCCWrapper
            primitive = SurfaceFactory.fromDict(deepcopy(feature))
            distances, angles = primitive.computeErrors(points, normals=normals,
                                                        symmetric_normals=ignore_primitives_orientation)
        except:
            pass
            print(f"WARNING: fail buiding a {feature['type']}.")    
    
    nan_mask = np.isnan(distances)
    distances[~nan_mask] /= reescale_factor_curr

    if np.any(nan_mask):
        #print(f"WARNING: nan distances in {feature['type']} geometry. Params: {feature}")    
        residual_distance = ResidualLoss()

        points[nan_mask, :], features_curr, _ = rescale(points, features=[feature], factor=1/reescale_factor_curr)
        feature = features_curr[0]
        distances[nan_mask] = residual_distance.residual_loss(points[nan_mask, :], feature)
    
    distance = np.nan if np.all(np.isnan(distances)) else np.nanmean(distances)*reescale_factor
    angle = np.nan if np.all(np.isnan(angles)) else np.nanmean(angles)

    return distance, angle

def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()

def process(data_tuple):
    if len(data_tuple) == 1:
        data = data_tuple[0]
        gt_data = None
    else:
        data, gt_data = data_tuple
        data = mergeQueryAndGTData(data, gt_data, force_match=force_match)

    filename = data['filename'] if 'filename' in data.keys() else str(i)
    points = data['noisy_points'] if use_noisy_points else data['points']
    normals = data['noisy_normals'] if use_noisy_normals else data['normals']

    labels = data['labels']
    labels[labels < -1] = -1 # removing non_gt_features

    features = data['features_data']
    if points is None or normals is None or labels is None or features is None:
        print('Invalid Model.')
        return None
    
    model_metrics = {}

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
        if gt_data is not None:
            _, _, reescale_factor = cubeRescale(gt_points.copy())
        else:
            _, _, reescale_factor = cubeRescale(points.copy())

    if gt_data is not None:
        model_major_diagonal = np.linalg.norm(np.max(gt_points, axis=0) - np.min(gt_points, axis=0))
    else:
        model_major_diagonal = np.linalg.norm(np.max(points, axis=0) - np.min(gt_points, axis=0))

    for i, feature in enumerate(features):
        indices = fpi[i]
        if feature is not None and indices is not None:
            points_curr = points[indices]
            normals_curr = normals[indices]    
            tp = feature['type']
                
            if tp not in model_metrics:
                model_metrics[tp] = get_base_metrics_dict(with_gt=(gt_data is not None))

            # Points
            model_metrics[tp]['n_prim_points'].append(len(indices))

            # Primitives (Boolean to work in individual models and in the entire dataset at the same time)
            model_metrics[tp]['n_prim'].append(True)
            if len(indices) == 0:
                model_metrics[tp]['n_void_prim'].append(True)
            elif 'invalid' in feature and feature['invalid']:
                model_metrics[tp]['n_invalid_prim'].append(True)

            # Distances (residual)
            if len(indices) > 0 and ('invalid' not in feature or not feature['invalid']):
                distance, angle = compute_deviations(points_curr, normals_curr, deepcopy(feature), reescale_factor=reescale_factor)

                invalid_primitive = (distance >= reescale_factor*model_major_diagonal)              

                if fpi_gt is not None:
                    indices_gt = fpi_gt[i]
                    points_gt_curr = gt_points[indices_gt]
                    normals_gt_curr = gt_normals[indices_gt]
                    
                    gt_distance, gt_angle = compute_deviations(points_gt_curr, normals_gt_curr, deepcopy(feature), reescale_factor=reescale_factor)

                    #invalid_primitive = invalid_primitive or (gt_distance >= reescale_factor*model_major_diagonal)

                    if not invalid_primitive:
                        model_metrics[tp]['gt_distance'].append(gt_distance)
                        model_metrics[tp]['gt_angle'].append(gt_angle)
            
                if not invalid_primitive:
                    model_metrics[tp]['distance'].append(distance)
                    model_metrics[tp]['angle'].append(angle)
                
                if invalid_primitive:
                    model_metrics[tp]['n_invalid_prim'].append(True)

            # IoUs (boolean for types to work in models and in the dataset)
            if len(indices) > 0:
                if gt_data is not None:
                    model_metrics[tp]['instance_iou'].append(instance_ious[i])
                    gt_tp = gt_data['features_data'][i]['type']
                    model_metrics[tp]['type_iou'].append(tp==gt_tp)

                if write_segmentation_gt:
                    colors_instances[indices, :] = computeRGB(colors_full[i%len(colors_full)])
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

    # Adding a key to compute the metrics agnostic of prim type
    model_metrics = generate_total_key_metrics_dict(model_metrics)
    model_metrics['Total']['n_no_prim_points'].append(np.count_nonzero(labels==-1))  # adding non primitivized points

    # Transforming from list to nd array each metric accumulator
    model_metrics = metrics_dict_list2array(model_metrics)

    derived_model_metrics = compute_derived_metrics(model_metrics)

    with open(f'{log_format_folder_name}/{filename}.json', 'w') as f:
        json.dump(derived_model_metrics, f, indent=4, default=np_encoder)
        
    if write_segmentation_gt:
        instances_filename = f'{filename}_instances.obj'
        writeColorPointCloudOBJ(join(seg_format_folder_name, instances_filename), np.concatenate((points, colors_instances), axis=1))
        types_filename = f'{filename}_types.obj'
        writeColorPointCloudOBJ(join(seg_format_folder_name, types_filename), np.concatenate((points, colors_types), axis=1))
    
    if box_plot:
        fig = generateErrorsBoxPlot(model_metrics)
        plt.figure(fig.number)
        plt.savefig(f'{box_plot_format_folder_name}/{filename}.png')
        plt.close()
        fig2 = generateErrorsBoxPlot(model_metrics, distances_key='gt_distance', angles_key='gt_angle')
        plt.figure(fig2.number)
        plt.savefig(f'{box_plot_format_folder_name}/{filename}_gt.png')
        plt.close()
    
    model_metrics_reduced = reduce_derived_model_metrics(derived_model_metrics)
    
    return model_metrics_reduced

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
    parser.add_argument('--no_use_occ', action='store_true')
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
    use_occ = not args['no_use_occ']
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

        dataset_metrics_dict = concatenate_metrics_dict(results)
        derived_dataset_metrics_dict = compute_derived_metrics(dataset_metrics_dict)

        if box_plot:
            fig = generateErrorsBoxPlot(dataset_metrics_dict, distances_key='distance', angles_key='angle')
            plt.figure(fig.number)
            plt.savefig(f'{box_plot_format_folder_name}/{s}.png')
            plt.close()
            fig2 = generateErrorsBoxPlot(dataset_metrics_dict, distances_key='gt_distance', angles_key='gt_angle')
            plt.figure(fig2.number)
            plt.savefig(f'{box_plot_format_folder_name}/{s}_to_gt.png')
            plt.close()

        with open(f'{log_format_folder_name}/{s}.json', 'w') as f:
            json.dump(derived_dataset_metrics_dict, f, indent=4, default=np_encoder)

        pprint(derived_dataset_metrics_dict)