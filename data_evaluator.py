import argparse
from pathlib import Path

import re
import h5py
import numpy as np
import pickle

from lib.utils import translateFeature
from lib.readers.spfn_dataset_reader import SpfnDatasetReader

def hungarian_matching(gt_mat, pred_mat):    
    gt_mat_arr = [np.array(gt) for gt in gt_mat]
    pred_mat_arr = [np.array(pred) for pred in pred_mat]

    for i in range(len(gt_mat_arr)):
        g = gt_mat_arr[i]
        g_t = g.transpose()
        p = pred_mat_arr[i]
        
        num = g_t*p
        den = np.linalg.norm(g, ord=2) + np.linalg.norm(p, ord=2) - num        

        print(num / den)


def open_h5_files(gt, pred):
    gt_file = h5py.File(gt)
    pred_file = h5py.File(pred)
    soup_filter = re.compile('(.*)_soup_([0-9]+)$')
    gt_matrices = []
    pred_matrices = []

    # For ground truth
    gt_noisy_points =   gt_file["noisy_points"][()] if 'noisy_points'   in gt_file.keys() else None
    gt_points =         gt_file['gt_points'][()]    if 'gt_points'      in gt_file.keys() else None
    gt_normals =        gt_file['gt_normals'][()]   if 'gt_normals'     in gt_file.keys() else None
    gt_labels =         gt_file['gt_labels'][()]    if 'gt_labels'      in gt_file.keys() else None
    
    gt_found_soup_ids = []
    gt_soup_id_to_key = {}
    for key in list(gt_file.keys()):
        m = soup_filter.match(key)
        if m is not None:
            gt_soup_id = int(m.group(2))
            gt_found_soup_ids.append(gt_soup_id)
            gt_soup_id_to_key[gt_soup_id] = key

    gt_features_data = []
    gt_found_soup_ids.sort()
    for i in range(len(gt_found_soup_ids)):
        g = gt_file[gt_soup_id_to_key[i]]

        gt_matrices.append(g["gt_points"][()])

        meta = pickle.loads(g.attrs['meta'])
        meta = translateFeature(meta, SpfnDatasetReader.FEATURES_BY_TYPE, SpfnDatasetReader.FEATURES_MAPPING)
        gt_features_data.append(meta)

    result_gt = {
        'noisy_points': gt_noisy_points,
        'points': gt_points,
        'normals': gt_normals,
        'labels': gt_labels,
        'features': gt_features_data
    }

    # For predict 
    pred_noisy_points =     pred_file["noisy_points"][()] if 'noisy_points'   in pred_file.keys() else None
    pred_points =           pred_file['gt_points'][()]    if 'gt_points'      in pred_file.keys() else None
    pred_normals =          pred_file['gt_normals'][()]   if 'gt_normals'     in pred_file.keys() else None
    pred_labels =           pred_file['gt_labels'][()]    if 'gt_labels'      in pred_file.keys() else None

    pred_found_soup_ids = []
    pred_soup_id_to_key = {}
    for key in list(pred_file.keys()):
        m = soup_filter.match(key)
        if m is not None:
            pred_soup_id = int(m.group(2))
            pred_found_soup_ids.append(pred_soup_id)
            pred_soup_id_to_key[pred_soup_id] = key

    pred_features_data = []
    pred_found_soup_ids.sort()
    for i in range(len(pred_found_soup_ids)):
        g = pred_file[pred_soup_id_to_key[i]]

        pred_matrices.append(g["gt_points"][()])

        meta = pickle.loads(g.attrs['meta'])
        meta = translateFeature(meta, SpfnDatasetReader.FEATURES_BY_TYPE, SpfnDatasetReader.FEATURES_MAPPING)
        pred_features_data.append(meta)

    result_pred = {
        'noisy_points': pred_noisy_points,
        'points': pred_points,
        'normals': pred_normals,
        'labels': pred_labels,
        'features': pred_features_data
    }

    hungarian_matching(gt_matrices, pred_matrices)

    return result_gt, result_pred

def list_files(input_dir: str, return_str: bool = False) -> list:
    files = []
    path = Path(input_dir)
    for file_path in path.glob('*'):
        if file_path.suffix.lower() == ".h5":
            files.append(file_path if not return_str else str(file_path))
    return sorted(files)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("input_dir", type=str, help="Directory of input data.")
    args = vars(parser.parse_args())

    input_directory = args["input_dir"]

    files = list_files(input_directory)
    for file in files:
        ground_truth, predict = open_h5_files(file, file)
        