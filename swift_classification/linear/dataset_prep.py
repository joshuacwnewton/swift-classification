import h5py
import numpy as np
from sklearn.model_selection import StratifiedKFold


def load_datasets(training_path, testing_path):
    """Extract training and testing data from provided .h5 files."""

    with h5py.File(training_path, "r") as h5_file_ptr:
        x_train = np.array(h5_file_ptr["data"])
        y_train = np.array(h5_file_ptr["label"])

    with h5py.File(testing_path, "r") as h5_file_ptr:
        x_test = np.array(h5_file_ptr["data"])
        y_test = np.array(h5_file_ptr["label"])

    return x_train, x_test, y_train, y_test


def apply_transforms(data, transforms):
    """Generic applier to apply transforms on a per-example basis.

    Args:
        -data: A set of examples to be iterated through.
        -transforms: A list of functions to apply to each example."""

    transformed_data = []
    for x in data:
        for transform in transforms:
            x = transform(x)

        transformed_data.append(x)

    return np.array(transformed_data)


def train_val_split(X, y, num_folds, cross_val=False):
    """Generator function that returns training and validation
    DataLoaders. If cross_val is specified, then each time the generator
    is called, a different fold will be designated as the validation
    fold."""

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True)

    for train_index, test_index in skf.split(X, y):
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]

        yield X_train, X_val, y_train, y_val

        if not cross_val:
            break
