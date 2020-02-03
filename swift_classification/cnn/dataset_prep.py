# stdlib imports
import copy

# dependency imports
import h5py
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import SubsetRandomSampler


class HDF5Dataset(data.Dataset):
    """Represents an abstract HDF5 dataset.

    Input params:
        file_path: Path to .h5 file. Must contain 'data' and 'label datasets.
        transform: PyTorch transforms to apply to every instance within the
        required 'data' dataset in the .h5 file. (default=None)
    """

    def __init__(self, file_path, transform=None):
        super().__init__()
        self.file_path = file_path
        self.transform = transform

        required_datasets = ["data", "label"]
        with h5py.File(self.file_path, "r") as h5_file_ptr:
            for ds in required_datasets:
                if ds not in h5_file_ptr.keys():
                    raise RuntimeError(f'File missing required "{ds}" dataset')

    def __len__(self):
        with h5py.File(self.file_path, "r") as h5_file_ptr:
            return h5_file_ptr["data"][()].shape[0]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        with h5py.File(self.file_path, "r") as h5_file_ptr:
            x = np.array(h5_file_ptr["data"][index])
            y = np.array(h5_file_ptr["label"][index])

        if isinstance(self.transform, list):
            for transform in self.transform:
                x = transform(x)
        elif self.transform:
            raise RuntimeError('Transform names must be provided by list.')

        return torch.from_numpy(x), torch.from_numpy(y)


def train_val_split(dataset, num_folds, loader_params, cross_val=False):
    """Generator function that returns training and validation
    DataLoaders. If cross_val is specified, then each time the generator
    is called, a different fold will be designated as the validation
    fold."""

    if cross_val:
        va_fold_nums = np.arange(num_folds)  # All available for validation
    else:
        va_fold_nums = np.arange(1)  # Use only the first fold for validation

    # Create folds containing shuffled indices
    all_folds = np.array(
        np.array_split(np.random.permutation(len(dataset)), num_folds)
    )

    # Create copy of loader params so "batch_size" can be overridden for val
    val_loader_params = copy.deepcopy(loader_params)

    for va_fold_num in va_fold_nums:
        # Use current fold for validation loader
        val_idxs = all_folds[va_fold_num]
        val_loader_params["batch_size"] = len(val_idxs)
        val_loader = data.DataLoader(dataset, **val_loader_params,
                                     sampler=SubsetRandomSampler(val_idxs))

        # Use all folds BUT current fold for training loader
        train_idxs = np.concatenate(
            all_folds[np.delete(np.arange(num_folds), va_fold_num)]
        )
        train_loader = data.DataLoader(dataset, **loader_params,
                                       sampler=SubsetRandomSampler(train_idxs))

        yield train_loader, val_loader
