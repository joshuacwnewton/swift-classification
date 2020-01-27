"""
    Utility script to pack raw, labeled images into HDF5 container.

    Requires images to be structured in the following format:

    <root directory>
        |-- class1
            |-- img1.png
            |-- img2.png
        |-- class2
            |-- img1.png
            |-- img2.png

"""

# Image loading imports
import os
import sys
import argparse
import glob
import re
import cv2

# Image splitting/packing imports
from sklearn.model_selection import train_test_split
import numpy as np
import h5py
from datetime import datetime


def main(args):
    X, y = load_images(args.input, encoding=".png")

    X_train, X_val, X_test, y_train, y_val, y_test \
        = split_dataset(X, y, train_size=args.split[0],
                        val_size=args.split[1],
                        test_size=args.split[2])

    # Initialize hdf5 file pointer
    f = h5py.File(f"{os.path.dirname(args.input)}/"
                  f"{datetime.now()}_{len(X)}.h5", "w")

    store_set(f, X_train, y_train, "training")
    store_set(f, X_val, y_val, "validation")
    store_set(f, X_test, y_test, "testing")

    f.close()


def load_images(root_dir, encoding=".png"):
    """Load all images (assumed to be stored in class subdirectories)
    into paired image/label sets."""
    filepaths = glob.glob(root_dir + "/**/*.png", recursive=True)

    # Choose pattern such that parent directory names are used as class labels
    pattern = re.compile(r"^.*\/(.+)\/[^\/]*.png$")
    matches = [pattern.match(f) for f in filepaths]

    images = [cv2.imread(m.group(0)) for m in matches if m is not None]
    images = [cv2.imencode(encoding, i)[1] for i in images]

    labels = [m.group(1) for m in matches if m is not None]

    return images, labels


def split_dataset(X, y, train_size, val_size, test_size):
    """Split overall dataset into stratified train/val/test sets based
    on provided percentage splits."""

    assert sum([train_size, val_size, test_size]) == 1.0

    # Split dataset into "validation set" and "the rest"
    X_rest, X_val, y_rest, y_val = train_test_split(X, y,
                                                    test_size=val_size,
                                                    train_size=(1 - val_size),
                                                    stratify=y)

    # Test size should be adjusted to be fraction of "the rest" subset
    test_count = round(test_size * len(X))
    test_size = test_count / len(X_rest)

    # Split "the rest" into "test set" and "training set"
    X_train, X_test, y_train, y_test = train_test_split(X_rest, y_rest,
                                                        test_size=test_size,
                                                        train_size=(1-test_size),
                                                        stratify=y_rest)

    assert len(X) == len(X_train) + len(X_val) + len(X_test)
    assert len(y) == len(y_train) + len(y_val) + len(y_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


def store_set(h5_fptr, data, labels, name):
    """Stores paired data and labels into passed h5 file pointer."""

    assert len(data) == len(labels)

    # Variable-length datatypes for encoded png streams and label names
    dt_int = h5py.vlen_dtype(np.dtype('uint8'))
    dt_str = h5py.string_dtype(encoding='utf-8')

    # Create group and store data/labels
    grp = h5_fptr.create_group(f'{name}_set')
    grp.create_dataset("data", (len(data),), dtype=dt_int, data=data)
    grp.create_dataset("label", data=np.array(labels, dtype=dt_str))


###############################################################################
#                ABOVE: IMAGE HANDLING | BELOW: ARG PARSING                   #
###############################################################################


class DirectoryAction(argparse.Action):
    """Checks if argument is a directory, and that it contains only
    subdirectories, before storing directory path."""

    def __call__(self, parser, namespace, values, option_string=None):
        if not os.path.isdir(values):
            raise NotADirectoryError("Specified directory does not exist.")

        for fname in os.listdir(values):
            if not os.path.isdir(os.path.join(values, fname)):
                raise AssertionError("Expected class subdirectories only"
                                     "within provided root directory.")

        setattr(namespace, self.dest, values)


class SplitAction(argparse.Action):
    """Checks if training/validation/testing percentages add to 100%
    before storing."""

    def __call__(self, parser, namespace, values, option_string=None):
        values = [float(value) for value in values]
        if sum(values) != 1.0:
            raise ValueError("Train/Val/Test percentages do not sum to 1.")
        setattr(namespace, self.dest, values)


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()

    parser.add_argument('--input',
                        help='Path to root directory containing image classes',
                        action=DirectoryAction)
    parser.add_argument('--split',
                        help="Percentage split between train/val/test",
                        nargs=3,
                        action=SplitAction)

    return parser.parse_args(args)


if __name__ is "__main__":
    args = parse_args()
    main(args)