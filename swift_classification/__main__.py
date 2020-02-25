from swift_classification.cnn.__main__ import main as cnn
from swift_classification.linear.__main__ import main as linear

from glob import glob
import argparse
from pathlib import Path
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(
        description='Configuration loader for classifiers'
    )

    subparsers = parser.add_subparsers(help='sub-command help')
    add_cnn_subparser(subparsers)
    add_linear_subparser(subparsers)

    transform_params = parser.add_argument_group("Dataset Transform Params")
    transform_params.add_argument(
        '--imread_mode',
        default=1,  # 1 == cv2.IMREAD_COLOR
        help='Number of subprocesses to use for data loading',
    )
    transform_params.add_argument(
        '--small_input_size',
        default=(24, 24),
        help='Size of image after resizing',
    )
    parser.add_argument(
        '--cross_val',
        action='store_true',
        help='Whether cross-validation should be performed'
    )

    parsed_args = parser.parse_args()

    # This is a quick fix to check out skorch, but if abandoning skorch, this
    # will need to be uncommented.
    # TODO: For reminder.
    # parsed_args = pack_loader_params(parsed_args)

    return parsed_args


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
        '--num_classes',
        default=2,
    )

    parser.add_argument(
        '--model',
        choices=['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet',
                 'inception'],
        default='squeezenet',
        help='Which pretrained CNN architecture to use'
    )
    parser.add_argument(
        '--input_size',
        default=(224, 224),
        help='Input size for specified CNN architecture',
    )
    parser.add_argument(
        '--feature_extract',
        action='store_true',
        help="Flag for feature extracting. When False, we fine-tune the whole "
             "model, when True we only update the reshaped layer params"
    )

    parser.add_argument(
        '--num_folds',
        default=5,
        help='Number of subportions for training/validation split'
    )
    parser.add_argument(
        '--num_epochs',
        default=14,
        help='Number of epochs to train classifier over'
    )

    loader_params = parser.add_argument_group("DataLoader Parameters")
    loader_params.add_argument(
        '--batch_size',
        default=8,
        help='Number of training samples per training batch',
    )
    loader_params.add_argument(
        '--num_workers',
        default=6,
        help='Number of subprocesses to use for data loading',
    )

    loader_params = parser.add_argument_group("Optimizer Parameters")
    loader_params.add_argument(
        '--learning_rate',
        default=0.001,
        help='Number of training samples per training batch',
    )
    loader_params.add_argument(
        '--momentum',
        default=0.9,
        help='Number of subprocesses to use for data loading',
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

    parser.add_argument(
        '--num_folds',
        default=5,
        help='Number of subportions for training/validation split'
    )
    parser.add_argument(
        '--num_epochs',
        default=100,
        help='Number of epochs to train classifier over'
    )

    loader_params = parser.add_argument_group("Normalization Parameters")
    loader_params.add_argument(
        '--mean',
        default=0,
        help='Number of training samples per training batch',
    )
    loader_params.add_argument(
        '--std',
        default=1,
        help='Number of subprocesses to use for data loading',
    )

    parser.set_defaults(main_func=linear)


###############################################################################
#                    /\ ARG PARSING | CUSTOM ACTIONS \/                       #
###############################################################################


def pack_loader_params(arguments):
    """Certain arguments are used solely in initializing a DataLoader
    object from the PyTorch library. Grouping them together here so they
    can be used by specifying **args.loader_params."""

    loader_params = {}
    for argname, value in vars(arguments).items():
        if argname in DataLoader.__init__.__code__.co_varnames:
            loader_params[argname] = value

    if len(loader_params) > 0:
        setattr(arguments, "loader_params", loader_params)
        for argname, value in loader_params.items():
            delattr(arguments, argname)

    return arguments


class FindHDF5InDir(argparse.Action):
    """Get paths for relevant h5 files stored within data directory."""

    def __call__(self, parser, namespace, values, option_string=None):
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
