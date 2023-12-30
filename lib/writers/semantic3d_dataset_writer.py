from .base_dataset_writer import BaseDatasetWriter
from lib.utils import filterFeaturesData, computeFeaturesPointIndices, filterFeaturesSemanticData

import os
import uuid
import pickle
import numpy as np

class Semantic3dDatasetWriter(BaseDatasetWriter):
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
        
        data_file_path = os.path.join(self.data_folder_name, f"{filename}.txt")
        labels_file_path = os.path.join(self.data_folder_name, f"{filename}_xyz_intensity_rgb.labels")
        transforms_file_path = os.path.join(self.transform_folder_name, f"{filename}.pkl")

        if type(features_data) == dict:
            features_data = features_data["surfaces"]
        
        if os.path.exists(data_file_path):
            return False

        self.filenames_by_set[self.current_set_name].append(filename)

        with open(labels_file_path, 'w') as labels_file:
            with open(data_file_path, 'w') as data_file:
                points, noisy_points, normals, noisy_normals, features_data, transforms = self.normalize(points, noisy_points, normals,
                                                                                                        noisy_normals, features_data)
                
                if np.any(np.isnan(points)) or np.any(np.isnan(normals)) or np.any(np.isnan(noisy_points)) or np.any(np.isnan(noisy_normals)):
                    print(np.any(np.isnan(points)), np.any(np.isnan(normals)), np.any(np.isnan(noisy_points)), np.any(np.isnan(noisy_normals)))
                
                with open(transforms_file_path, 'wb') as pkl_file:
                    pickle.dump(transforms, pkl_file)

                intensity_rgb = "0 0 0 0 0 0\n"
                for point_idx, point in enumerate(points):
                    x = str(point[0])
                    y = str(point[1])
                    z = str(point[2])
                    line = x + ' ' + y + ' ' + z + ' ' + intensity_rgb

                    data_file.write(line)

                    the_labels_point_idx = semantic_labels[point_idx]
                    obj = semantic_data[the_labels_point_idx]
                    label = obj["label"]
                    label_id = str(Semantic3dDatasetWriter.BASE_CLASSES[label]) + '\n'
                    
                    labels_file.write(label_id)
                    
        if not os.path.exists(data_file_path):
            assert False
        return True
    
    def finish(self, permutation=None):
        train_models, val_models = self.divisionTrainVal(permutation=permutation)

        with open(os.path.join(self.data_folder_name, 'train_data.txt'), 'w') as f:
            f.write('\n'.join(train_models))
        with open(os.path.join(self.data_folder_name, 'val_data.txt'), 'w') as f:
            f.write('\n'.join(val_models))
        with open(os.path.join(self.data_folder_name, 'test_data.txt'), 'w') as f:
            f.write('\n'.join(val_models))

        super().finish()

