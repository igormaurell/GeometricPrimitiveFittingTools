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
from lib.utils import computeFeaturesPointIndices, writeColorPointCloudOBJ, getAllColorsArray, computeRGB

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
    print(data_distances)
    if len(data_distances) > 0:
        ax1.boxplot(data_distances, labels=data_labels, autorange=False, meanline=True)
    ax2.set_title('Normal Deviation')
    print(data_angles)
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
    for tp in errors.keys():
        number_of_primitives = len(errors[tp]['distances'])
        number_of_void_primitives = len(errors[tp]['void_primitives'])
        total_number_of_primitives += number_of_primitives
        total_number_of_void_primitives += number_of_void_primitives
        log += f'\n\t- {tp}:\n'
        log += f'\t\t- Number of Primitives: {number_of_primitives + number_of_void_primitives}\n'
        log += f'\t\t- Number of Void Primitives: {number_of_void_primitives}\n'

        ind_distances = errors[tp]['distances']
        ind_angles = errors[tp]['distances']
        number_of_points = 0
        number_error_distances = 0
        number_error_angles = 0
        for i in range(len(ind_distances)):
            number_of_points += len(ind_distances[i])
            for elem in ind_distances[i]:
                if elem > max_distance_deviation:
                    number_error_distances += 1
            for elem in ind_angles[i]:
                if elem > max_angle_deviation:
                    number_error_angles += 1
        total_number_of_points += number_of_points
        total_number_error_distances += number_error_distances
        total_number_error_angles += number_error_angles

        log += f'\t\t- Distance Error Rate (<{max_distance_deviation}): {(100.0*number_error_distances)/number_of_points}%\n'
        log += f'\t\t- Normal Error Rate (<{max_angle_deviation}): {(100.0*number_error_angles)/number_of_points}%\n'

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
    log_total += f'\t\t- Distance Error Rate (<{max_distance_deviation}): {(100.0*total_number_error_distances)/total_number_of_points}%\n'
    log_total += f'\t\t- Normal Error Rate (<{max_angle_deviation}): {(100.0*total_number_error_angles)/total_number_of_points}%\n'
    log_total += f'\t\t- Mean Distance Error: {total_mean_distances/total_number_of_primitives}\n'
    log_total += f'\t\t- Mean Normal Error: {total_mean_angles/total_number_of_primitives}\n\n'

    return log_total + log


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
            labels_type = np.zeros(len(labels), dtype=np.int64) - 1

            dataset_errors[filename] = {}

            log = printAndLog(f'\n-- File {filename}:\n', log)

            fpi = computeFeaturesPointIndices(labels, size=len(features))
            for i, feature in tqdm(enumerate(features)):
                points_curr = points[fpi[i]]
                normals_curr = normals[fpi[i]]
                primitive = PrimitiveSurfaceFactory.primitiveFromDict(feature)
                errors = primitive.computeErrors(points_curr, normals_curr)
                tp = primitive.getPrimitiveType()

                tp_label = PrimitiveSurfaceFactory.getTypeLabel(tp)
                labels_type[fpi[i]] = tp_label
                
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
                instances_filename = f'{filename}_instances.obj'
                colors = getAllColorsArray()
                labels_color = np.zeros(shape=points.shape)
                for i in range(len(labels_color)):
                    labels_color[i, :] = computeRGB(colors[labels[i]%len(colors)])
                writeColorPointCloudOBJ(join(seg_format_folder_name, instances_filename), np.concatenate((points, labels_color), axis=1))

                types_filename = f'{filename}_types.obj'
                colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,00), (255,255,255)]
                labels_color = np.zeros(shape=points.shape)
                for i in range(len(labels_color)):
                    labels_color[i, :] = colors[labels_type[i]]
                writeColorPointCloudOBJ(join(seg_format_folder_name, types_filename), np.concatenate((points, labels_color), axis=1))

            error_log = generateErrorsLog(dataset_errors[filename], max_distance_deviation, max_normal_deviation)

            log = printAndLog(error_log, log)
            with open(f'{log_format_folder_name}/{filename}.txt', 'w') as f:
                f.write(log)

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

            
                
    exit()
    for h5_filename in h5_files if VERBOSE else tqdm(h5_files):
        report = ''
        report+= f'\n-- Processing file {h5_filename}...\n'
        if VERBOSE:
            print(f'\n-- Processing file {h5_filename}...')

        bar_position = h5_filename.rfind('/')
        point_position = h5_filename.rfind('.')
        base_filename = h5_filename[(bar_position+1):point_position]

        h5_file = readH5(join(h5_folder_name, h5_filename))

        if write_segmentation_gt:
            labels_type = createLabelsByType(h5_file)

            points = h5_file['gt_points']
            labels = h5_file['gt_labels']

            instances_filename = f'{base_filename}_instances.obj'
            writeSegmentedPointCloudOBJ(join(seg_folder_name, instances_filename), points, labels)

            types_filename = f'{base_filename}_types.obj'
            colors = [(255,255,255), (255,0,0), (0,255,0), (0,0,255), (255,255,0)]
            writeSegmentedPointCloudOBJ(join(seg_folder_name, types_filename), points, labels_type, colors=colors)
        
        error_results = calculateError(h5_file)

        number_of_primitives = 0
        distance_error = 0
        normal_dev_error = 0
        mean_distance = 0.
        mean_normal_dev = 0.
        for key in error_results.keys():
            number_of_primitives += len(error_results[key]['mean_distance'])
            distance_error += np.count_nonzero(error_results[key]['mean_distance'] > max_distance_deviation)
            normal_dev_error += np.count_nonzero(error_results[key]['mean_normal'] > max_normal_deviation)
            mean_distance += np.sum(error_results[key]['mean_distance'])
            mean_normal_dev += np.sum(error_results[key]['mean_normal'])
        
        if number_of_primitives > 0:
            distance_error /= number_of_primitives
            normal_dev_error /= number_of_primitives
            mean_distance /= number_of_primitives
            mean_normal_dev /= number_of_primitives
        
        report+= f'{h5_filename} is processed.\n'
        if VERBOSE:
            print(f'{h5_filename} is processed.')

        report+= '\nTESTING REPORT:\n'
        report+= '\n- Total:\n'
        report+= f'\t- Number of Primitives: {number_of_primitives}\n' 
        report+= f'\t- Distance Error Rate: {(distance_error)*100} %\n'
        report+= f'\t- Normal Error Rate: {(normal_dev_error)*100} %\n'
        report+= f'\t- Mean Distance Error: {mean_distance}\n'
        report+= f'\t- Mean Normal Error: {mean_normal_dev}\n' 

        print('\nTESTING REPORT:')
        print('\n- Total:')
        print('\t- Number of Primitives:', number_of_primitives)
        print('\t- Distance Error Rate:', (distance_error)*100, '%')
        print('\t- Normal Error Rate:', (normal_dev_error)*100, '%')
        print('\t- Mean Distance Error:', mean_distance)
        print('\t- Mean Normal Error:', mean_normal_dev)
        data_distance = []
        data_normal = []
        labels = []
        for key in error_results.keys():
            name = key[0].upper() + key[1:]
            number_of_primitives_loc = len(error_results[key]['mean_distance'])
            distance_error_rate = (np.count_nonzero(error_results[key]['mean_distance'] > max_distance_deviation)/number_of_primitives_loc)*100
            normal_error_rate = (np.count_nonzero(error_results[key]['mean_normal'] > max_normal_deviation)/number_of_primitives_loc)*100
            distance_error = np.mean(error_results[key]['mean_distance'])
            normal_error = np.mean(error_results[key]['mean_normal'])

            report+= f'\n- {name}:\n'
            report+= f'\t- Number of Primitives: {number_of_primitives_loc}\n' 
            report+= f'\t- Distance Error Rate: {distance_error_rate} %\n'
            report+= f'\t- Normal Error Rate: {normal_error_rate} %\n'
            report+= f'\t- Mean Distance Error: {distance_error}\n'
            report+= f'\t- Mean Normal Error: {normal_error}\n' 

            print(f'\n- {name}:')
            print('\t- Number of Primitives:', number_of_primitives_loc)
            print('\t- Distance Error Rate:', distance_error_rate, '%')
            print('\t- Normal Error Rate:', normal_error_rate, '%')
            print('\t- Mean Distance Error:', distance_error)
            print('\t- Mean Normal Error:', normal_error)
            data_distance.append(error_results[key]['mean_distance'])
            data_normal.append(error_results[key]['mean_normal'])
            labels.append(name)

        with open(join(log_folder_name, f'{base_filename}.txt'), 'w') as f:
            f.write(report)
        if box_plot:
            fig, (ax1, ax2) = plt.subplots(2, 1)
            fig.tight_layout(pad=2.0)
            ax1.set_title('Distance Deviation')
            if len(data_distance) > 0:
                ax1.boxplot(data_distance, labels=labels, autorange=False, meanline=True)
            ax2.set_title('Normal Deviation')
            if len(data_distance) > 0:
                ax2.boxplot(data_normal, labels=labels, autorange=False, meanline=True)
            plt.savefig(join(box_plot_folder_name, f'{base_filename}.png'))
            plt.show(block=False)
            plt.pause(10)
            plt.close()
        