from .base_dataset_writer import BaseDatasetWriter
from lib.utils import computeSemanticPointLabels

import os
import uuid
import pickle
import numpy as np

class S3DISDatasetWriter(BaseDatasetWriter):
    BASE_CLASSES = {
        "unlabeled": 0,
        "tank": 1,
        "pipe": 2,
        "silo": 3,
        "instrumentation": 4,
        "floor": 5,
        "wall": 6,
        "structure": 7
    }

    def __init__(self, parameters):
        super().__init__(parameters)

    def step(self, points, normals=None, labels=None, features_data=[], noisy_points=None,
            noisy_normals=None, filename=None, features_point_indices=None, semantic_data=[], semantic_labels=[], semantic_point_indices=[], **kwargs):

        if filename is None:
            filename = str(uuid.uuid4())

        data_dir_path = os.path.join(self.data_folder_name, f"Area_{filename}")
        os.makedirs(data_dir_path, exist_ok=True)

        data_file_path = os.path.join(data_dir_path, f"{filename}.txt")
        transforms_file_path = os.path.join(self.data_folder_name, f"{filename}.pkl")
        
        labels_dir_path = os.path.join(data_dir_path, "Annotations")
        os.makedirs(labels_dir_path, exist_ok=True)

        if type(features_data) == dict:
            features_data = features_data["surfaces"]

        if os.path.exists(data_file_path):
            return False
        
        if not semantic_point_indices:
            semantic_point_indices, semantic_point_labels = computeSemanticPointLabels(semantic_labels, semantic_data, size=len(semantic_data))
        
        self.filenames_by_set[self.current_set_name].append(filename)

        with open(data_file_path, 'w') as data_file:
            points, noisy_points, normals, noisy_normals, features_data, transforms = self.normalize(points, noisy_points, normals,
                                                                                                        noisy_normals, features_data)

            if np.any(np.isnan(points)) or np.any(np.isnan(normals)) or np.any(np.isnan(noisy_points)) or np.any(np.isnan(noisy_normals)):
                print(np.any(np.isnan(points)), np.any(np.isnan(normals)), np.any(np.isnan(noisy_points)), np.any(np.isnan(noisy_normals)))

            with open(transforms_file_path, 'wb') as pkl_file:
                pickle.dump(transforms, pkl_file)
                
            for point_idx, point in enumerate(points):
                x = str(point[0])
                y = str(point[1])
                z = str(point[2])
                point_line = x + " " + y + " " + z + " " + "0 0 0\n" # rgb

                data_file.write(point_line)

            for key, value in semantic_point_labels.items():
                label_file_path = os.path.join(labels_dir_path, f"{key}.txt")
                with open(label_file_path, 'w') as label_file:
                    for point_idx in value:
                        point = points[point_idx]

                        x = str(point[0])
                        y = str(point[1])
                        z = str(point[2])
                        point_line = x + " " + y + " " + z + " " + "0 0 0\n" # rgb

                        label_file.write(point_line)

    def finish(self, permutation=None):
        train_models, val_models = self.divisionTrainVal(permutation=permutation)

        with open(os.path.join(self.data_folder_name, 'train_data.txt'), 'w') as f:
            f.write('\n'.join(train_models))
        with open(os.path.join(self.data_folder_name, 'val_data.txt'), 'w') as f:
            f.write('\n'.join(val_models))
        with open(os.path.join(self.data_folder_name, 'test_data.txt'), 'w') as f:
            f.write('\n'.join(val_models))

        super().finish()
