import os
import uuid

from .base_dataset_writer import BaseDatasetWriter

# For normal estimation
class PcpnetDatasetWriter(BaseDatasetWriter):

    def __init__(self, parameters):
        super().__init__(parameters)

    def step(self, points, normals, noisy_points=None, noisy_normals=None, filename=None, **kwargs):
        if filename is None:
            filename = str(uuid.uuid4())

        filename = filename.strip()

        xyz_file_path     = os.path.join(self.data_folder_name, f"{filename}.xyz")
        normals_file_path = os.path.join(self.data_folder_name, f"{filename}.normals")

        if os.path.exists(xyz_file_path):
            print(f"INFO: {xyz_file_path} already exist.")
            return False
        
        if os.path.exists(normals_file_path):
            print(f"INFO: {normals_file_path} already exist.")
            return False

        self.filenames_by_set[self.current_set_name].append(filename)

        points, noisy_points, normals, noisy_normals, _, _ = self.normalize(points, noisy_points, normals, noisy_normals, [])

        with open(xyz_file_path, "w") as xyz_file:
            for noisy_point in noisy_points:
                xyz_file.write("{} {} {}\n".format(noisy_point[0], noisy_point[1], noisy_point[2]))
            
        with open(normals_file_path, "w") as normals_file:
            for noisy_normal in noisy_normals:
                normals_file.write("{} {} {}\n".format(noisy_normal[0], noisy_normal[1], noisy_normal[2]))

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

