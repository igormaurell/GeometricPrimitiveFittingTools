import pickle
import h5py
from os.path import join
import re

from .base_dataset_reader import BaseDatasetReader

class SpfnDatasetReader(BaseDatasetReader):

    def __init__(self, parameters):
        super().__init__(parameters)

        with open(join(self.data_folder_name, 'train_models.csv'), 'r') as f:
            text = f.read()
            self.filenames_by_set['train'] = text.split(', ')
        with open(join(self.data_folder_name, 'test_models.csv'), 'r') as f:
            text = f.read()
            self.filenames_by_set['test'] = text.split(', ')

    def step(self):
        assert self.current_set_name in self.filenames_by_set.keys()

        index = self.steps_by_set[self.current_set_name]%len(self.filenames_by_set[self.current_set_name])
        filename = self.filenames_by_set[self.current_set_name][index]
        point_position = filename.rfind('.')

        data_file_path = join(self.data_folder_name, filename)
        filename = filename[:point_position]
        transforms_file_path = join(self.transform_folder_name, f'{filename}.pkl')

        with h5py.File(data_file_path, 'r') as h5_file:
            noisy_points = h5_file['noisy_points'][()] if 'noisy_points' in h5_file.keys() else None
            gt_points = h5_file['gt_points'][()] if 'gt_points' in h5_file.keys() else None
            gt_normals = h5_file['gt_normals'][()] if 'gt_normals' in h5_file.keys() else None
            labels = h5_file[()] if 'labels' in h5_file.keys() else None

            found_soup_ids = []
            soup_id_to_key = {}
            soup_prog = re.compile('(.*)_soup_([0-9]+)$')
            for key in list(h5_file.keys()):
                m = soup_prog.match(key)
                if m is not None:
                    soup_id = int(m.group(2))
                    found_soup_ids.append(soup_id)
                    soup_id_to_key[soup_id] = key

            features_data = []            
            found_soup_ids.sort()
            for i in range(len(found_soup_ids)):
                g = h5_file[soup_id_to_key[i]]
                meta = pickle.loads(g.attrs['meta'])
                features_data.append(meta)

        result = {
            'noisy_points': noisy_points,
            'points': gt_points,
            'normals': gt_normals,
            'labels': labels,
            'features': features_data,
        }

        self.steps_by_set[self.current_set_name] += 1
        
        return result
    
    def finish(self):
        super().finish()