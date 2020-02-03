# dependency imports
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# project imports
from .dataset_prep import load_datasets, apply_transforms, train_val_split
from ..preprocessing import Decode, Resize, HOG


def main(args):
    x_trainval, x_test, y_trainval, y_test = load_datasets(args.train_path,
                                                           args.test_path)

    transforms = [Decode(flags=args.imread_mode),
                  Resize(output_size=args.resize_dim),
                  HOG()]
    x_trainval = apply_transforms(x_trainval, transforms)

    for x_train, x_val, y_train, y_val in train_val_split(x_trainval,
                                                          y_trainval,
                                                          args.num_folds,
                                                          args.cross_val):
        # Normalize the data based on training
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_val = scaler.transform(x_val)

        # Fit training data to SVM classifier with RBF
        clf = SVC()
        clf.fit(x_train, y_train)

        # Get predictions and evaluate
        y_pred = clf.predict(x_val)
        cm = confusion_matrix(y_val, y_pred)

        test = None

    x_test = apply_transforms(x_test, transforms)