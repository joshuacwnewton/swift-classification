from .dataset_prep import load_datasets, train_val_split


def main(args):
    train_set, test_set = load_datasets(args.train_path, args.test_path)

    loader_params = {"batch_size": args.batch_size,
                     "num_workers": args.num_workers}
    for train_loader, val_loader in train_val_split(train_set,
                                                    args.num_folds,
                                                    loader_params,
                                                    args.cross_val):
        for i_epoch in range(args.num_epochs):
            for x_batch, y_batch in train_loader:
                pass

            x_val, y_val = next(iter(val_loader))
