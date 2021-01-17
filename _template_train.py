import joblib
import random

import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# pip install python-mnist
from mnist import MNIST
# download 4 MNIST files you can here:
# http://yann.lecun.com/exdb/mnist/


MNIST_CELL_SIZE = 28
SEED = 0xBadCafe


def fix_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(save_model_path='random_forest.joblib', mnist_data_path='mnist_data', print_accuracy=False):

    # FIXING RANDOM SEED:
    fix_seed(SEED)

    # EXAMPLE OF LOADING MNIST
    mnist_data = MNIST(mnist_data_path)
    mnist_data.gz = True

    images_train, labels_train = mnist_data.load_training()
    images_test, labels_test = mnist_data.load_testing()

    images_train = np.uint8([np.reshape(im, (MNIST_CELL_SIZE,) * 2) for im in images_train])
    images_test = np.uint8([np.reshape(im, (MNIST_CELL_SIZE,) * 2) for im in images_test])
    labels_train, labels_test = np.int16(labels_train), np.int16(labels_test)

    # EXAMPLE OF TRAINING RandomForest

    # hog = get_hog()  # from cv2
    # features_train = np.array([hog.compute(im).T[0] for im in images_train])
    # features_test = np.array([hog.compute(im).T[0] for im in images_test])
    features_train = np.array([im.ravel() for im in images_train])
    features_test = np.array([im.ravel() for im in images_test])

    rf = RandomForestClassifier(n_jobs=-1, random_state=SEED)
    rf.fit(features_train, labels_train)

    # SAVING MODEL !!!
    joblib.dump(rf, save_model_path)
    joblib.load()

    if print_accuracy:
        from sklearn.metrics import accuracy_score
        print(accuracy_score(labels_test, rf.predict(features_test)))
