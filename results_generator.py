import argparse
from os.path import join
from os import makedirs
from shutil import rmtree
from tqdm import tqdm
import numpy as np
import open3d as o3d

from asGeometryOCCWrapper.surfaces import SurfaceFactory

from lib.readers import DatasetReaderFactory
from lib.utils import computeRGB, getAllColorsArray, saveFeatures, computeFeaturesPointIndices

def computeAnnotationPosition(points):
    mean = np.mean(points, axis=0)
    closest = np.argmin(np.linalg.norm(points - mean, axis=1))
    return points[closest].tolist()


def featuresToAnnotations(features_data, points):
    annotations = []
    counts = {}
    for i, feature in enumerate(features_data):
        tp = feature["type"]
        if tp not in counts:
            counts[tp] = 0

        annotation = {}
        annotation['title'] = f'{tp} {counts[tp]}'
        counts[tp] += 1
        annotation['description'] = f'This is a {tp} primitive.'
        annotation['position'] = computeAnnotationPosition(points[np.asarray(feature['point_indices'])])
        annotations.append(annotation)
    
    return {'annotations': annotations}

def generateVisualResults(features_data, points):
    type_results = {'types': o3d.geometry.PointCloud()}
    instance_results = {'instances': o3d.geometry.PointCloud()}

    colors_full = getAllColorsArray()

    used_points = np.zeros(len(points)).astype(np.bool)

    for i, feature in enumerate(features_data):
        instance_rgb = np.asarray(computeRGB(colors_full[i%len(colors_full)])).astype(np.float32)/255
        type_rgb = np.asarray(SurfaceFactory.FEATURES_SURFACE_CLASSES[feature['type']].getColor()).astype(np.float32)/255

        tp = feature['type'].lower()

        points_curr = points[np.asarray(feature['point_indices'])]
        used_points[np.asarray(feature['point_indices'])] = True

        instance_cloud = o3d.geometry.PointCloud()
        instance_cloud.points = o3d.utility.Vector3dVector(points_curr)
        instance_cloud.paint_uniform_color(instance_rgb)
        instance_results['instances'] += instance_cloud
        instance_results[str(i)] = instance_cloud

        if tp not in type_results:
            type_results[tp] = o3d.geometry.PointCloud()

        type_cloud = o3d.geometry.PointCloud()
        type_cloud.points = o3d.utility.Vector3dVector(points_curr)
        type_cloud.paint_uniform_color(type_rgb)
        type_results['types'] += type_cloud
        type_results[tp] += type_cloud
    
    unlabeled_points = points[np.logical_not(used_points)]
    unlabeled_cloud = o3d.geometry.PointCloud()
    unlabeled_cloud.points = o3d.utility.Vector3dVector(unlabeled_points)
    unlabeled_cloud.paint_uniform_color([0, 0, 0])

    type_results['types'] += unlabeled_cloud
    instance_results['instances'] += unlabeled_cloud
    type_results['unlabeled'] = unlabeled_cloud

    return {'types': type_results, 'instances': instance_results}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('folder', type=str, help='dataset folder.')
    formats_txt = ','.join(DatasetReaderFactory.READERS_DICT.keys())
    parser.add_argument('format', type=str, help=f'types of h5 format to generate. Possible formats: {formats_txt}. Multiple formats can me generated.')
    parser.add_argument('output_folder', type=str, help='output folder.')

    parser.add_argument('--dataset_folder_name', type=str, default = 'dataset', help='input dataset folder name.')
    parser.add_argument('--data_folder_name', type=str, default = 'data', help='data folder name.')
    parser.add_argument('--transform_folder_name', type=str, default = 'transform', help='transform folder name.')

    args = vars(parser.parse_args())

    folder_name = args['folder']
    input_format = args['format']
    output_folder_name = args['output_folder']
    dataset_folder_name = args['dataset_folder_name']
    data_folder_name = args['data_folder_name']
    transform_folder_name = args['transform_folder_name']

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
    parameters[input_format]['unnormalize'] = True

    dataset_reader_factory = DatasetReaderFactory(parameters)
    reader = dataset_reader_factory.getReaderByFormat(input_format)

    rmtree(output_folder_name, ignore_errors=True)

    makedirs(output_folder_name, exist_ok=True)

    colors = getAllColorsArray()

    reader.setCurrentSetName('val')

    for i, data in enumerate(tqdm(reader)):
        features_point_indices = computeFeaturesPointIndices(data['labels'], len(data['features_data']))

        for j in range(len(data['features_data'])):
            data['features_data'][j]['point_indices'] = features_point_indices[j].tolist()

        saveFeatures(join(output_folder_name, f"{data['filename']}_primitives"),
                     data['features_data'], tp='yaml')

        saveFeatures(join(output_folder_name, f"{data['filename']}_primitives"),
                     data['features_data'], tp='json')

        annotations = featuresToAnnotations(data['features_data'], data['points'])
        saveFeatures(join(output_folder_name, f"{data['filename']}_annotations"),
                     annotations, tp='json')
        
        visual_results_dict = generateVisualResults(data['features_data'], data['points'])

        for key1 in visual_results_dict.keys():
            for key2 in visual_results_dict[key1].keys():
                o3d.io.write_point_cloud(join(output_folder_name, f"{data['filename']}_{key2}.ply"),
                                         visual_results_dict[key1][key2])
