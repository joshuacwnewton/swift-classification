from .dataset_prep import HDF5Dataset, train_val_split
from swift_classification.image_processing import Decode, Resize


def main(args):
    transforms = [Decode(flags=args.imread_mode),
                  Resize(output_size=args.resize_dim)]
    train_set = HDF5Dataset(args.train_path, transforms)
    test_set = HDF5Dataset(args.test_path, transforms)

    for train_loader, val_loader in train_val_split(train_set,
                                                    args.num_folds,
                                                    args.loader_params,
                                                    args.cross_val):
        for i_epoch in range(args.num_epochs):
            for x_batch, y_batch in train_loader:
                pass

            x_val, y_val = next(iter(val_loader))
