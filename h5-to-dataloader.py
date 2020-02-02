# import helpers
import numpy as np
from pathlib import Path
import torch
from torch.utils import data
import cv2
from skimage import transform
from glob import glob
import h5py

from transforms import Decode, Resize


def main(training_path, testing_path):
    # TODO: Split config off into separate functions
    num_epochs = 50
    loader_params = {'batch_size': 100, 'shuffle': True, 'num_workers': 6}

    # Transforms to be applied across datasets
    decoder = Decode(flags=cv2.IMREAD_COLOR)
    resizer = Resize(output_size=24)
    dataset = HDF5Dataset('data/', recursive=True, load_data=False,
                          data_cache_size=4, transform=[decoder, resizer])

    data_loader = data.DataLoader(dataset, **loader_params)

    for i in range(num_epochs):
        for x, y in data_loader:
            # here comes your training loop
            pass


###############################################################################
#              ABOVE: TRAINING/TESTING | BELOW: DATASET CLASS                 #
###############################################################################


class HDF5Dataset(data.Dataset):
    """Represents an abstract HDF5 dataset.

    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """

    def __init__(self, file_path, recursive, load_data, data_cache_size=3,
                 transform=None):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        self.transform = transform

        # Search for all h5 files
        p = Path(file_path)
        assert (p.is_dir())
        if recursive:
            files = sorted(p.glob('**/*.h5'))
        else:
            files = sorted(p.glob('*.h5'))
        if len(files) < 1:
            raise RuntimeError('No hdf5 datasets found')

        for h5dataset_fp in files:
            self._add_data_infos(str(h5dataset_fp.resolve()), load_data)

    def __getitem__(self, index):
        # get data
        x = self.get_data("data", index)
        if isinstance(self.transform, list):
            for transform in self.transform:
                x = transform(x)
        elif self.transform:
            raise RuntimeError('Transform names must be provided by list.')
        else:
            x = torch.from_numpy(x)

        # get label
        y = self.get_data("label", index)
        y = torch.from_numpy(y)
        return (x, y)

    def __len__(self):
        return len(self.get_data_infos('data'))

    def _add_data_infos(self, file_path, load_data):
        with h5py.File(file_path) as h5_file:
            # Walk through all groups, extracting datasets
            for gname, group in h5_file.items():
                for dname, ds in group.items():
                    # if data is not loaded its cache index is -1
                    idx = -1
                    if load_data:
                        # add data to the data cache
                        idx = self._add_to_cache(ds.value, file_path)

                    # type is derived from the name of the dataset; we expect the dataset
                    # name to have a name such as 'data' or 'label' to identify its type
                    # we also store the shape of the data in case we need it
                    self.data_info.append(
                        {'file_path': file_path, 'type': dname,
                         'shape': ds.value.shape, 'cache_idx': idx})

    def _load_data(self, file_path):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        with h5py.File(file_path) as h5_file:
            for gname, group in h5_file.items():
                for dname, ds in group.items():
                    # add data to the data cache and retrieve
                    # the cache index
                    idx = self._add_to_cache(ds.value, file_path)

                    # find the beginning index of the hdf5 file we are looking for
                    file_idx = next(i for i, v in enumerate(self.data_info) if
                                    v['file_path'] == file_path)

                    # the data info should have the same index since we loaded it in the same way
                    self.data_info[file_idx + idx]['cache_idx'] = idx

        # remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.data_cache_size:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path)
            self.data_cache.pop(removal_keys[0])
            # remove invalid cache_idx
            self.data_info = [
                {'file_path': di['file_path'], 'type': di['type'],
                 'shape': di['shape'], 'cache_idx': -1} if di['file_path'] ==
                                                           removal_keys[
                                                               0] else di for
                di in self.data_info]

    def _add_to_cache(self, data, file_path):
        """Adds data to the cache and returns its index. There is one cache
        list for every file_path, containing all datasets in that file.
        """
        if file_path not in self.data_cache:
            self.data_cache[file_path] = [data]
        else:
            self.data_cache[file_path].append(data)
        return len(self.data_cache[file_path]) - 1

    def get_data_infos(self, type):
        """Get data infos belonging to a certain type of data.
        """
        data_info_type = [di for di in self.data_info if di['type'] == type]
        return data_info_type

    def get_data(self, type, i):
        """Call this function anytime you want to access a chunk of data from the
            dataset. This will make sure that the data is loaded in case it is
            not part of the data cache.
        """
        fp = self.get_data_infos(type)[i]['file_path']
        if fp not in self.data_cache:
            self._load_data(fp)

        # get new cache_idx assigned by _load_data_info
        cache_idx = self.get_data_infos(type)[i]['cache_idx']
        return self.data_cache[fp][cache_idx]


###############################################################################
#                 ABOVE: DATASET CLASS | BELOW: ARG PARSING                   #
###############################################################################


def find_files(directory, name, ext, recursive=False):

    if recursive:
        files = sorted(glob(f'{directory}/**/*{name}*.{ext}'))
    else:
        files = sorted(glob(f'{directory}/*{name}*.{ext}'))

    if len(files) < 1:
        raise RuntimeError('No hdf5 datasets found')

    return files


if __name__ == "__main__":
    # TODO: Use proper argparsing once functionality in place
    data_dir = "data/current_*"

    h5_train = find_files(data_dir, "train", "h5")
    h5_test = find_files(data_dir, "test", "h5")

    main(h5_train, h5_test)





