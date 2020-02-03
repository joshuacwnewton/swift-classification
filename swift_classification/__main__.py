from swift_classification.cnn.__main__ import main as cnn
from swift_classification.linear.__main__ import main as linear

from glob import glob
import argparse
import sys
from pathlib import Path


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description='Configuration loader for classifiers'
    )
    subparsers = parser.add_subparsers(help='sub-command help')

    add_cnn_subparser(subparsers)
    add_linear_subparser(subparsers)

    return parser.parse_args(args)


def add_cnn_subparser(subparsers):
    """Subparser containing configuration options for a CNN-based
    classification model."""

    parser = subparsers.add_parser('cnn', help='Use CNN-based classifier')
    parser.add_argument(
        'data_dir',
        help='The directory containing train/test datasets',
        action=FindHDF5InDir
    )
    parser.add_argument(
        '--num_folds',
        default=5,
        help='Number of subportions for training/validation split'
    )
    parser.add_argument(
        '--cross_val',
        action='store_false',
        help='Whether cross-validation should be performed'
    )
    parser.add_argument(
        '--num_epochs',
        default=100,
        help='Number of epochs to train classifier over'
    )
    parser.add_argument(
        '--batch_size',
        default=100,
        help='DataLoader parameter: how many samples per training batch'
    )
    parser.add_argument(
        '--num_workers',
        default=6,
        help='DataLoader parameter: # of subprocesses to use for data loading.'
    )

    parser.set_defaults(main_func=cnn)


def add_linear_subparser(subparsers):
    """Subparser containing configuration options for a linear
    classification model (in this case, SVM)."""

    parser = subparsers.add_parser('linear', help='Use linear classifier')
    parser.add_argument(
        'data_dir',
        help='The directory containing train/test datasets',
        action=FindHDF5InDir
    )

    parser.set_defaults(main_func=linear)


###############################################################################
#                /\ ARG PARSING | CUSTOM DIRECTORY ACTION \/                  #
###############################################################################


class FindHDF5InDir(argparse.Action):
    """Get paths for relevant h5 files stored within data directory."""

    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values, list):
            raise RuntimeError('Only one data directory may be passed.')

        training_dataset = find_n_files(values, "train", "h5", n=1)
        testing_dataset = find_n_files(values, "test", "h5", n=1)

        delattr(namespace, "data_dir")
        setattr(namespace, "train_path", training_dataset)
        setattr(namespace, "test_path", testing_dataset)


def find_n_files(directory, name, ext, n, recursive=True):
    """Return first n files as defined by basic search query.

    Args:
        -directory: path to search for files
        -name: string which filename should contain
        -ext: extension of file (without leading '.')
        -n: number of files to return, error thrown if fewer than n
        -recursive: whether subdirectories should be searched.
    """
    if n < 1:
        raise RuntimeError("N must be a positive integer.")

    directory = Path(directory).resolve()

    if recursive:
        files = sorted(glob(f'{directory}/**/*{name}*.{ext}', recursive=True))
    else:
        files = sorted(glob(f'{directory}/*{name}*.{ext}'))

    if len(files) < n:
        raise RuntimeError(f'Fewer than {n} {ext} datasets '
                           f'with name {name} found.')
    if n == 1:
        return files[0]
    else:
        return files[:n]


###############################################################################
#               /\ CUSTOM DIRECTORY ACTION | MAIN FUNCTION \/                 #
###############################################################################


if __name__ == "__main__":
    args = parse_args()
    args.main_func(args)  # Which main function is called depends on subparser
