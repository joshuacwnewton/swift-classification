# dependency imports
import h5py
import numpy as np
import torch
from torch.utils import data


class HDF5Dataset(data.Dataset):
    """Represents an abstract HDF5 dataset.

    Input params:
        file_path: Path to .h5 file. Must contain 'data' and 'label datasets.
        transform: PyTorch transforms to apply to every instance within the
        required 'data' dataset in the .h5 file. (default=None)
    """

    def __init__(self, file_path, transform=None, subset=None):
        super().__init__()
        self.file_path = file_path
        self.transform = transform
        self.subset_indices = subset

        required_datasets = ["data", "label"]
        with h5py.File(self.file_path, "r") as h5_file_ptr:
            for ds in required_datasets:
                if ds not in h5_file_ptr.keys():
                    raise RuntimeError(f'File missing required "{ds}" dataset')

    def __len__(self):
        if self.subset_indices is not None:
            ds_length = len(self.subset_indices)
        else:
            with h5py.File(self.file_path, "r") as h5_file_ptr:
                ds_length = h5_file_ptr["data"][()].shape[0]

        return ds_length

    def __getitem__(self, index):
        if self.subset_indices is not None:
            index = self.subset_indices[index]

        with h5py.File(self.file_path, "r") as h5_file_ptr:
            x = np.array(h5_file_ptr["data"][index])
            y = np.array(h5_file_ptr["label"][index])

        if isinstance(self.transform, list):
            for transform in self.transform:
                x = transform(x)
        elif self.transform:
            raise RuntimeError('Transform names must be provided by list.')

        y = torch.from_numpy(y)
        if len(y) > 1:
            y = y[0]

        return x, y


def train_val_idx_split(h5_path, num_folds, cross_val=False):
    """Generator function that returns training and validation
    indexes. If cross_val is specified, then each time the generator
    is called, a different fold will be designated as the validation
    fold."""

    with h5py.File(h5_path, "r") as h5_file_ptr:
        length = h5_file_ptr["data"][()].shape[0]

    if cross_val:
        va_fold_nums = np.arange(num_folds)  # All available for validation
    else:
        va_fold_nums = np.arange(1)  # Use only the first fold for validation

    # Create folds containing shuffled indices
    all_folds = np.array(
        np.array_split(np.random.permutation(length), num_folds)
    )

    for va_fold_num in va_fold_nums:
        # Use current fold for validation loader
        val_idxs = all_folds[va_fold_num]

        # Use all folds BUT current fold for training loader
        train_idxs = np.concatenate(
            all_folds[np.delete(np.arange(num_folds), va_fold_num)]
        )

        yield train_idxs, val_idxs
