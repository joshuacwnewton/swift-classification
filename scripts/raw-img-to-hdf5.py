"""
    Utility script to pack raw, labeled images into HDF5 container.
    Requires corresponding csv file with labels to pack.
"""

# Image/label loading imports
import os
import sys
import argparse
import glob
import re
import cv2
from pathlib import Path
import pandas as pd
from natsort import natsorted

# Image splitting/packing imports
from sklearn.model_selection import train_test_split
import numpy as np
import h5py
from datetime import datetime


def main(args):
    # Create unique directory based on time so .h5 files not overwritten
    output_dir = Path(f"{os.path.dirname(args.input)}"
                      f"/datasets/{datetime.now()}")
    Path.mkdir(output_dir, parents=True)

    # Load images as matrices and as encoded bytestreams
    X, X_encoded, filenames = load_data(f"{args.input}/images")

    # Load corresponding labesl and metadata from csv files
    metadata, y = load_labels(args.input, filenames, args.classes)

    # Split dataset and export to HDF5 files
    X_train, X_test, y_train, y_test = split_dataset(X_encoded, y,
                                                     train_size=args.split[0],
                                                     test_size=args.split[1])
    export_set(output_dir, "training", X_train, y_train, classes=args.classes)
    export_set(output_dir, "testing", X_test, y_test, classes=args.classes)

    # Export images to class folders for visualization
    if args.visual:
        visualize_images(output_dir, X, y, metadata)


def load_data(data_dir, encoding=".png"):
    """Load images in two forms: matrix form, and encoded bytestream."""

    filepaths = natsorted(glob.glob(f"{data_dir}/*.png"))

    # Choose pattern such that parent directory names are used as class labels
    pattern = re.compile(r"^.+\/([^\/]+.png)$")
    matches = [pattern.match(f) for f in filepaths]

    images = [cv2.imread(m.group(0)) for m in matches if (m is not None)]
    encoded_images = [cv2.imencode(encoding, i)[1] for i in images]

    filenames = [m.group(1) for m in matches if (m is not None)]

    return images, encoded_images, filenames


def load_labels(root_dir, filenames, classes):
    """Load labels as one-hot integer vectors from specified csv
    file. Also treat first two rows as metadata."""

    csv_filepath = glob.glob(f"{root_dir}/*.csv")[0]
    df_labels = pd.read_csv(csv_filepath, index_col="filename")

    assert len(filenames) == len(df_labels)

    # Assume any missing values are no by default
    df_labels = df_labels.fillna("no")

    for filename in filenames:
        # Get series corresponding to image file
        row_label = df_labels.loc[filename]

        # Convert label strings ('no', 'yes', etc.) into integers
        # ('0', '1', etc.) using mapping
        row_num = np.array([classes[l][v] if (l in classes.keys()) else v
                            for l, v in row_label.iteritems()])

        # Store integer labels into new dataframe
        df_labels.loc[filename] = row_num

    metadata = df_labels.values[:, :2]
    df_labels = df_labels.drop(columns=["src_vid", "segment_id"]).astype(int)

    return metadata, df_labels


def split_dataset(X, y, train_size, test_size):
    """Split overall dataset into stratified train/val/test sets based
    on provided percentage splits."""

    assert sum([train_size, test_size]) == 1.0

    # Split dataset into "validation set" and "the rest"
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        train_size=train_size)

    assert len(X) == len(X_train) + len(X_test)
    assert len(y) == len(y_train) + len(y_test)

    return X_train, X_test, y_train, y_test


def export_set(output_dir, name, data, labels, classes):
    """Stores paired data and labels into passed h5 file pointer."""

    assert len(data) == len(labels)

    # Variable-length datatypes for encoded png streams and label names
    dt_int = h5py.vlen_dtype(np.dtype('uint8'))
    dt_str = h5py.string_dtype(encoding='utf-8')

    # Initialize hdf5 file pointer
    f = h5py.File(f"{output_dir}/{name}_{len(data)}.h5", "w")

    # Create group and store data/labels
    x = f.create_dataset("data", (len(data),), dtype=dt_int, data=data)
    y = f.create_dataset("label", data=np.array(labels, dtype=int))

    # Store <mapping from (0, 1 ...) to class names> as group attribute
    y.attrs.create("class_names", data=np.array(classes, dtype=dt_str))

    f.close()


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
            if ((not Path(f"{values}/{fname}").is_dir())
                 and (Path(fname).suffix != ".csv")):
                raise RuntimeError("Directory must only contain subdirectories"
                                   " and csv files.")

        setattr(namespace, self.dest, values)


class SplitAction(argparse.Action):
    """Checks if training/validation/testing percentages add to 100%
    before storing."""

    def __call__(self, parser, namespace, values, option_string=None):
        values = [float(value) for value in values]
        if sum(values) != 1.0:
            raise ValueError("Train/Val and Test percentages do not sum to 1.")
        setattr(namespace, self.dest, values)


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()

    parser.add_argument('--input',
                        help='Path to root directory containing image classes',
                        action=DirectoryAction)
    parser.add_argument('--split',
                        help="Percentage split between train+val and test",
                        nargs=2,
                        action=SplitAction)
    parser.add_argument("--classes",
                        help="Config for class mapping. Don't input manually!",
                        default={
                            "keep": {"no": 0, "yes": 1},
                            "swift": {"no": 0, "1": 1, "2": 2, "3+": 3},
                            "blurry": {"no": 0, "yes": 1},
                            "chimney": {"no": 0, "yes": 1},
                            "antennae": {"no": 0, "yes": 1},
                            "non-swift": {"no": 0, "crow": 1,
                                          "seagull": 2, "other": 3}
                        })
    parser.add_argument("--visual",
                        help="Flag to export class visualizations",
                        action="store_true")

    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
