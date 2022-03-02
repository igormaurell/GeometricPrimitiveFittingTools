import argparse
from os.path import join, exists, isfile
from os import listdir, mkdir
from shutil import rmtree
from math import sqrt, acos, pi
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from lib.dataset_reader_factory import DatasetReaderFactory

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Geometric Primitive Fitting Results, works for dataset validation and for methods results')
    parser.add_argument('folder', type=str, help='dataset folder.')
    formats_txt = ','.join(DatasetReaderFactory.MAKERS_DICT.keys())
    parser.add_argument('format', type=str, help=f'types of h5 format to generate. Possible formats: {formats_txt}. Multiple formats can me generated.')    parser.add_argument('--h5_folder_name', type=str, default = 'h5', help='h5 folder name.')
    parser.add_argument('--dataset_folder_name', type=str, default = 'dataset', help='input dataset folder name.')
    parser.add_argument('--data_folder_name', type=str, default = 'data', help='data folder name.')
    parser.add_argument('--result_folder_name', type=str, default = 'val', help='validation folder name.')
    parser.add_argument('-v', '--verbose', action='store_true', help='show more verbose logs.')
    parser.add_argument('-s', '--segmentation_gt', action='store_true', help='write segmentation ground truth.')
    parser.add_argument('-b', '--show_box_plot', action='store_true', help='show box plot of the data.')
    parser.add_argument('-md', '--max_distance_deviation', type=float, default=0.5, help='max distance deviation.')
    parser.add_argument('-mn', '--max_normal_deviation', type=float, default=10, help='max normal deviation.')

    args = vars(parser.parse_args())

    folder_name = args['folder']
    dataset_folder_name = args['input_dataset_folder_name']
    format = args['input_format']
    data_folder_name = args['data_folder_name']
    result_folder_name = args['result_folder_name']
    VERBOSE = args['verbose']
    write_segmentation_gt = args['segmentation_gt']
    box_plot = args['show_box_plot']
    max_distance_deviation = args['max_distance_deviation']
    max_normal_deviation = args['max_normal_deviation']*pi/180.

    data_format_folder_name = join(folder_name, dataset_folder_name, format, data_folder_name)
    vol_

 
    if exists(h5_folder_name):
        h5_files = sorted([f for f in listdir(h5_folder_name) if isfile(join(h5_folder_name, f))])
    else:
        if VERBOSE:
            print('\nThere is no h5 folder.\n')
        exit()
    
    if exists(result_folder_name):
        rmtree(result_folder_name)
    mkdir(result_folder_name)

    seg_folder_name = join(result_folder_name, 'seg')

    if write_segmentation_gt:
        if exists(seg_folder_name):
            rmtree(seg_folder_name)
        mkdir(seg_folder_name)

    box_plot_folder_name = join(result_folder_name, 'boxplot')
    
    if box_plot:
        if exists(box_plot_folder_name):
            rmtree(box_plot_folder_name)
        mkdir(box_plot_folder_name)

    log_folder_name = join(result_folder_name, 'log')
    
    if exists(log_folder_name):
        rmtree(log_folder_name)
    mkdir(log_folder_name)


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
        