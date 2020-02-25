import copy

from .dataset_prep import HDF5Dataset, train_val_split
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, \
    balanced_accuracy_score

import torch
from torch.utils import data
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from swift_classification.preprocessing import Decode  # Custom transform

from skorch import NeuralNetClassifier
from skorch.helper import SliceDataset
from skorch.callbacks import LRScheduler, Checkpoint, Freezer
from sklearn.model_selection import GridSearchCV

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(args):
    # Step 1: Fetch Datasets for training/validation/test datasets
    train_set, val_set, test_set = get_datasets(args)

    # Step 2: Initialize callbacks for NeuralNetClassifier
    lrscheduler = LRScheduler(
        policy='StepLR', step_size=7, gamma=0.1)

    checkpoint = Checkpoint(
        f_params='best_model.pt', monitor='valid_acc_best')

    freezer = Freezer(lambda x: not x.startswith('model.classifier.1'))

    net = NeuralNetClassifier(
        TwoClassSqueezeNet,
        criterion=nn.CrossEntropyLoss,
        batch_size=args.batch_size,
        max_epochs=args.num_epochs,
        module__num_classes=args.num_classes,
        optimizer=optim.SGD,
        iterator_train__shuffle=True,
        iterator_train__num_workers=args.num_workers,
        iterator_valid__shuffle=True,
        iterator_valid__num_workers=args.num_workers,
        # train_split fixes bug in skorch library, see:
        # https://github.com/skorch-dev/skorch/issues/599
        train_split=None,
        device='cuda'  # comment to train on cpu
    )

    params = {
            'optimizer__lr': [1e-5, 1e-4, 1e-3],
            'optimizer__momentum': [0.5, 0.9, 0.99, 0.999],
            'optimizer__nesterov': [True, False]
        }
    gs = GridSearchCV(net, params, refit=False, cv=3, scoring='accuracy',
                      verbose=10)

    X_sl = SliceDataset(train_set, idx=0)  # idx=0 is the default
    y_sl = SliceDataset(train_set, idx=1)

    # net.fit(train_set, y=None)
    gs.fit(X_sl, y_sl)
    print(gs.best_score_, gs.best_params_)


class TwoClassSqueezeNet(nn.Module):
    """Rewritten setup_model() using object-oriented approach."""

    def __init__(self, num_classes):
        super().__init__()

        # Replace (512, 1000) output layer with (512, num_classes) layer
        model = models.squeezenet1_0(pretrained=True)
        num_ftrs = model.classifier[1].in_channels
        model.classifier[1] = nn.Conv2d(num_ftrs, num_classes, kernel_size=1)
        model.num_classes = num_classes
        model = model.to(device)

        self.model = model

    def forward(self, x):
        return self.model(x)


def get_datasets(args):
    """Hacky copy/paste to get just Datasets (not DataLoaders).

    I would typically never copy and paste like this, but I am trying to
    quickly figure out the distinctions between skorch's API and
    PyTorch's, so quick iteration is helpful.

    Original code in Pure PyTorch section below."""

    # Initialize image transforms for Dataset class
    data_transforms = [
        Decode(flags=args.imread_mode),
        transforms.ToPILImage(),
        transforms.Resize(args.small_input_size),
        transforms.Pad((args.input_size[0] - args.small_input_size[0])//2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    # Get indices for training/validation split -> Dataset
    train_idxs, val_idxs = next(train_val_split(args.train_path,
                                                args.num_folds,
                                                args.cross_val))
    train_set = HDF5Dataset(args.train_path, data_transforms)
                            # subset=train_idxs)
    val_set = HDF5Dataset(args.train_path, data_transforms, subset=val_idxs)

    # Use full test dataset for testing -> Dataset
    test_set = HDF5Dataset(args.test_path, data_transforms)
    # args.loader_params["batch_size"] = 1  # No minibatches for testing

    return train_set, val_set, test_set


###############################################################################
#               OLDER (PURE PYTORCH) SQUEEZENET CODE BELOW.                   #
###############################################################################


def old_boilerplate_main(args):
    # Step 1: Initialize
    model, params_to_update = setup_model(args)
    optimizer = setup_optimizer(args, params_to_update)
    criterion = setup_criterion(args)

    # Step 2: Fetch DataLoaders for training/validation/test datasets
    train_loader, val_loader, test_loader = get_data_loaders(args)

    # Step 3: Find best trained model over a number of epochs
    best_model = copy.deepcopy(model.state_dict())
    best_metric = 0.0
    for epoch in range(args.num_epochs):
        # Step 3.a: Train model.
        model, train_metric = train_model(model, optimizer, criterion,
                                          train_loader)
        print(f"Epoch {epoch} |  Training set  | "
              f"Balanced accuracy: {train_metric}")

        # Step 3.b: Validate trained model, update if best.
        model, val_metric = validate_model(model, optimizer, criterion,
                                           val_loader)
        print(f"Epoch {epoch} | Validation set | "
              f"Balanced accuracy: {val_metric}")
        if val_metric > best_metric:
            best_metric = val_metric
            best_model = copy.deepcopy(model.state_dict())

        print("--------------------------------------------------------------")

    # Step 4: Test best model on training dataset
    model.load_state_dict(best_model)
    test_model(model, test_loader)


def setup_model(args):
    """Select CNN architecture, modified for transfer learning on
    specific dataset."""
    # Call model constructor
    model = models.squeezenet1_0(pretrained=True)

    # Freeze layer parameters if feature extracting
    if args.feature_extract:
        for param in model.parameters():
            param.requires_grad = False

    # Modify model to support fewer classes: (512, 1000) -> (512, 2)
    # This will also unfreeze this layer's parameters, as the default value for
    # (weight/bias).required_grad = True
    model.classifier[1] = nn.Conv2d(512, args.num_classes, kernel_size=1)
    model.num_classes = args.num_classes

    # Send the model to GPU
    model = model.to(device)

    # Create list of parameters that should be updated (must be done after
    # freezing, and after last layer is reshaped).
    params_to_update = [p for p in model.parameters()
                        if p.requires_grad is True]

    return model, params_to_update


def setup_optimizer(args, params_to_update):
    """Select optimizer to modify weights based on losses."""

    optimizer = optim.SGD(params_to_update, lr=args.learning_rate,
                          momentum=args.momentum)

    return optimizer


def setup_criterion(args):
    """Select loss function for evaluating model predictions."""

    criterion = nn.CrossEntropyLoss()

    return criterion


def get_data_loaders(args):
    """Get dataloaders for training/validation/test split."""

    # Initialize image transforms for Dataset class
    data_transforms = [
        Decode(flags=args.imread_mode),
        transforms.ToPILImage(),
        transforms.Resize(args.small_input_size),
        transforms.Pad((args.input_size[0] - args.small_input_size[0])//2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    # Get indices for training/validation split -> Dataset -> DataLoader
    train_idxs, val_idxs = next(train_val_split(args.train_path,
                                                args.num_folds,
                                                args.cross_val))
    train_set = HDF5Dataset(args.train_path, data_transforms,
                            subset=train_idxs)
    val_set = HDF5Dataset(args.train_path, data_transforms, subset=val_idxs)
    train_loader = data.DataLoader(train_set, **args.loader_params)
    val_loader = data.DataLoader(val_set, **args.loader_params)

    # Use full test dataset for testing -> Dataset -> DataLoader
    test_set = HDF5Dataset(args.test_path, data_transforms)
    args.loader_params["batch_size"] = 1  # No minibatches for testing
    test_loader = data.DataLoader(test_set, **args.loader_params)

    return train_loader, val_loader, test_loader


def train_model(model, optimizer, criterion, train_loader):
    """Train model using training set."""

    # Prepare model and autograd for training
    model.train()

    # Iterate through mini-batches in training DataLoader
    y, y_pred = [], []
    for X_batch, y_batch in train_loader:
        # Send the inputs and labels to GPU
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            # Calculate output scores and loss for the batch
            scores_batch = model(X_batch)
            loss_batch = criterion(scores_batch, y_batch)

            # Generate predictions
            _, y_pred_batch = torch.max(scores_batch, 1)

            # Apply backpropagation
            loss_batch.backward()

            # Update the optimizer
            optimizer.step()

        # Append batch values to epoch tracker
        y.append(y_batch.cpu().numpy())
        y_pred.append(y_pred_batch.cpu().numpy())

    # Convert list of Tensors to single numpy arrays
    y = np.concatenate(y)
    y_pred = np.concatenate(y_pred)

    # Compute performance metric for training set
    train_metric = balanced_accuracy_score(y, y_pred)

    return model, train_metric


def validate_model(model, optimizer, criterion, val_loader):
    """Validate model using validation set."""
    # Prepare model and autograd for validation
    model.eval()

    # Iterate through mini-batches in testing DataLoader
    X, y, y_pred, loss = [], [], [], []
    for X_batch, y_batch in val_loader:
        # Send the inputs and labels to GPU
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        with torch.set_grad_enabled(False):
            # Calculate output scores and loss
            scores_batch = model(X_batch)
            loss_batch = criterion(scores_batch, y_batch)

            # Generate predictions
            _, y_pred_batch = torch.max(scores_batch, 1)

        # Append batch values to epoch tracker
        X.append(X_batch)
        y.append(y_batch)
        y_pred.append(y_pred_batch)
        loss.append(loss_batch)

    # Convert list of Tensors to single numpy arrays
    y = torch.cat(y).cpu().numpy()
    y_pred = torch.cat(y_pred).cpu().numpy()

    # Compute performance metric for validation set
    val_metric = balanced_accuracy_score(y, y_pred)

    return model, val_metric


def test_model(model, test_loader):
    """Test model using testing set."""

    X, y, y_pred = [], [], []
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Calculate output scores
        scores_batch = model(X_batch)

        # Generate predictions
        _, y_pred_batch = torch.max(scores_batch, 1)

        # Append batch values to test tracker
        X.append(X_batch)
        y.append(y_batch)
        y_pred.append(y_pred_batch)

    print("Results for test set: ")
    print(confusion_matrix(y, y_pred))
    print(classification_report(y, y_pred))
































































