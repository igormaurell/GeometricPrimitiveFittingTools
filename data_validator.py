import argparse
from os.path import join
from os import makedirs
from shutil import rmtree
from math import pi
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from lib.primitive_surface_factory import PrimitiveSurfaceFactory
from lib.dataset_reader_factory import DatasetReaderFactory
from lib.utils import sortedIndicesIntersection, computeFeaturesPointIndices, writeColorPointCloudOBJ, getAllColorsArray, computeRGB

def printAndLog(text, log):
    print(text)
    log += text
    return log

def generateErrorsBoxPlot(errors, individual=True, all_models=False):
    data_distances = []
    data_angles = []
    data_labels = []
    if all_models:
        pass
    else:
        for tp, e in errors.items():
            data_labels.append(tp)
            data_distances.append(np.concatenate(e['distances']))
            data_angles.append(np.concatenate(e['angles']))
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.tight_layout(pad=2.0)
    ax1.set_title('Distance Deviation')
    if len(data_distances) > 0:
        ax1.boxplot(data_distances, labels=data_labels, autorange=False, meanline=True)
    ax2.set_title('Normal Deviation')
    if len(data_angles) > 0:
        ax2.boxplot(data_angles, labels=data_labels, autorange=False, meanline=True)
    return fig

def generateErrorsLog(errors, max_distance_deviation, max_angle_deviation, save_path=None):
    log = ''
    total_number_of_primitives = 0
    total_number_of_void_primitives = 0
    total_number_of_points = 0
    total_number_error_distances = 0
    total_number_error_angles = 0
    total_mean_distances = 0.
    total_mean_angles = 0.
    for tp, e in errors.items():
        number_of_primitives = len(e['distances'])
        number_of_void_primitives = len(e['void_primitives'])
        total_number_of_primitives += number_of_primitives
        total_number_of_void_primitives += number_of_void_primitives
        log += f'\n\t- {tp}:\n'
        log += f'\t\t- Number of Primitives: {number_of_primitives + number_of_void_primitives}\n'
        log += f'\t\t- Number of Void Primitives: {number_of_void_primitives}\n'

        ind_distances = e['distances']
        ind_angles = e['angles']
        number_of_points = 0
        number_error_distances = 0
        number_error_angles = 0
        for i in range(len(ind_distances)):
            number_of_points += len(ind_distances[i])
            number_error_distances += np.count_nonzero(ind_distances[i] > max_distance_deviation)
            number_error_angles += np.count_nonzero(ind_angles[i] > max_angle_deviation)

        total_number_of_points += number_of_points
        total_number_error_distances += number_error_distances
        total_number_error_angles += number_error_angles


        log += f'\t\t- Distance Error Rate (>{max_distance_deviation}): {(100.0*number_error_distances)/number_of_points}%\n'
        log += f'\t\t- Normal Error Rate (>{max_angle_deviation}): {(100.0*number_error_angles)/number_of_points}%\n'

        mean_distances = np.array(errors[tp]['mean_distances'])
        summd = np.sum(mean_distances)
        total_mean_distances += summd
        mean_angles = np.array(errors[tp]['mean_angles'])
        summa = np.sum(mean_angles)
        total_mean_angles += summa
        log += f'\t\t- Mean Distance Error: {summd/number_of_primitives}\n'
        log += f'\t\t- Mean Normal Error: {summa/number_of_primitives}\n\n'

    log_total = f'\n\t- Total:\n'
    log_total += f'\t\t- Number of Primitives: {total_number_of_primitives + total_number_of_void_primitives}\n'
    log_total += f'\t\t- Number of Void Primitives: {total_number_of_void_primitives}\n'
    log_total += f'\t\t- Distance Error Rate (>{max_distance_deviation}): {(100.0*total_number_error_distances)/total_number_of_points}%\n'
    log_total += f'\t\t- Normal Error Rate (>{max_angle_deviation}): {(100.0*total_number_error_angles)/total_number_of_points}%\n'
    log_total += f'\t\t- Mean Distance Error: {total_mean_distances/total_number_of_primitives}\n'
    log_total += f'\t\t- Mean Normal Error: {total_mean_angles/total_number_of_primitives}\n\n'

    return log_total + log

def computeErrorsArrays(indices, distances, angles, max_distance_deviation, max_angle_deviation):
    error_dist = np.sort(indices[distances > max_distance_deviation])
    error_ang = np.sort(indices[angles > max_angle_deviation])
    return error_dist, error_ang

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
    parser.add_argument('-md', '--max_distance_deviation', type=float, default=50, help='max distance deviation.')
    parser.add_argument('-mn', '--max_normal_deviation', type=float, default=10, help='max normal deviation.')

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
    max_normal_deviation = args['max_normal_deviation']*pi/180.

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
    sets = ['test', 'train']
    colors_full = getAllColorsArray()
    for s in sets:
        full_log = ''
        reader.setCurrentSetName(s)
        print(f'\nValidation of {s} dataset:')
        dataset_errors = {}
        for i in range(len(reader)):
            log = ''
            
            data = reader.step()

            filename = data['filename'] if 'filename' in data.keys() else str(i)
            points = data['points']
            normals = data['normals']
            labels = data['labels']
            features = data['features']

            dataset_errors[filename] = {}

            log = printAndLog(f'\n-- File {filename}:\n', log)

            colors_instances = np.zeros(shape=points.shape, dtype=np.int64) + np.array([255, 255, 255])
            colors_types = np.zeros(shape=points.shape, dtype=np.int64) + np.array([255, 255, 255])

            fpi = computeFeaturesPointIndices(labels, size=len(features))
            for i, feature in tqdm(enumerate(features)):
                points_curr = points[fpi[i]]
                normals_curr = normals[fpi[i]]
                primitive = PrimitiveSurfaceFactory.primitiveFromDict(feature)
                errors = primitive.computeErrors(points_curr, normals_curr)
                tp = primitive.getPrimitiveType()
                
                if tp not in dataset_errors[filename]:
                    dataset_errors[filename][tp] = {'distances': [], 'mean_distances': [], 'angles': [], 'mean_angles': [], 'void_primitives': []}

                if len(fpi[i]) == 0:
                    dataset_errors[filename][tp]['void_primitives'].append(i)
                else:
                    dataset_errors[filename][tp]['distances'].append(errors['distances'])
                    dataset_errors[filename][tp]['angles'].append(errors['angles'])
                    dataset_errors[filename][tp]['mean_distances'].append(np.mean(errors['distances']))
                    dataset_errors[filename][tp]['mean_angles'].append(np.mean(errors['angles']))
                
                    if write_segmentation_gt:
                        colors_instances[fpi[i], :] = computeRGB(colors_full[i%len(colors_full)])
                        colors_types[fpi[i], :] = primitive.getColor()
                        if write_points_error:
                            error_dist, error_ang = computeErrorsArrays(fpi[i], errors['distances'], errors['angles'], max_distance_deviation, max_normal_deviation)
                            error_both = sortedIndicesIntersection(error_dist, error_ang)

                            colors_instances[error_dist, :] = np.array([0, 255, 255])
                            colors_types[error_dist, :] = np.array([0, 255, 255])

                            colors_instances[error_ang, :] = np.array([0, 0, 0])
                            colors_types[error_ang, :] = np.array([0, 0, 0])

                            colors_instances[error_both, :] = np.array([255, 0, 255])
                            colors_types[error_both, :] = np.array([255, 0, 255])

            error_log = generateErrorsLog(dataset_errors[filename], max_distance_deviation, max_normal_deviation)
            log = printAndLog(error_log, log)
            with open(f'{log_format_folder_name}/{filename}.txt', 'w') as f:
                f.write(log)

            if write_segmentation_gt:
                instances_filename = f'{filename}_instances.obj'
                
                writeColorPointCloudOBJ(join(seg_format_folder_name, instances_filename), np.concatenate((points, colors_instances), axis=1))

                types_filename = f'{filename}_types.obj'
                writeColorPointCloudOBJ(join(seg_format_folder_name, types_filename), np.concatenate((points, colors_types), axis=1))

            if box_plot:
                fig = generateErrorsBoxPlot(dataset_errors[filename])
                plt.figure(fig.number)
                plt.savefig(f'{box_plot_format_folder_name}/{filename}.png')
                plt.show(block=False)
                plt.pause(10)
                plt.close()

            full_log += log
        
        with open(f'{log_format_folder_name}/{s}.txt', 'w') as f:
            f.write(full_log)