#! /usr/bin/env python2.7

import sys
sys.path.append("../util/")
import argparse
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
import pywt
import cv2
import datetime
import scipy.cluster.vq
import scipy.misc
import scipy.ndimage
import matplotlib.pyplot as plt
import skimage.feature
import sklearn.cross_validation
import sklearn.metrics
import models.cloud_classifier
import models.svm as svm
import models.mlp as mlp
import utils

# Set how numpy handles errors
np.seterr(all='ignore')


DEFAULT_CLASS_SIZE = 100
DEFAULT_OUTPUT_PATH = '../../data/models/'
DEFAULT_CLASS_MAP = '../../data/cloud-types.csv'
DEFAULT_K_FOLD = 10


def main():
    # Get the arguments
    args = parse_arguments()

    # Get the training parameters
    class_size = args.class_size if args.class_size is not None else DEFAULT_CLASS_SIZE
 
    # Get the train and dev feature files
    train_files = args.train_files
    dev_files = args.dev_files

    # Create the list of models to train
    models = []
    if args.svm: models.append(svm.SVM())
    if args.mlp: models.append(mlp.MLP(10))

    # If there no development files, perform cross-validation
    if dev_files == None:
        train_data, train_labels = utils.read_features(train_files)

        models = train_cross_validation(train_data, train_labels, models, class_size)
    # Otherwise use the development files
    else:
        train_data, train_labels = utils.read_features(train_files)
        dev_data, dev_labels = utils.read_features(dev_files)

        train(train_data, train_labels, dev_data, dev_labels)


def parse_arguments():
    """ 
        This function retrieves the command line parameters passed into this script.

        Parameters
        ----------
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-files', type=str, nargs='+', help='List the names of the training feature files')
    parser.add_argument('--dev-files', type=str, nargs='+', help='List the names of the development feature files')
    parser.add_argument('--class-size', type=int, help='The number of examples to use per class')
    parser.add_argument('--svm', action='store_true', help='Specify to enable SVM classifier')
    parser.add_argument('--mlp', action='store_true', help='Specify to enable MLP ANN classifier')
    args = parser.parse_args()

    return args


def train_cross_validation(data, labels, models, class_size, k_fold=DEFAULT_K_FOLD):
    """ 
        This function trains some models and tests them with cross validation.

        Parameters
        ----------
        data:
            the training data
        labels:
            the corresponding labels for the training data
        models:
            the models to train
        class_size:
            the amount of examples to use per class
    """
    def gen_svm_params():
        params = {}
        params['degree'] = 0
        params['gamma'] = 1
        params['coef'] = 0
        params['C'] = 1
        params['nu'] = 0
        params['p'] = 0
        params['svm_type'] = cv2.ml.SVM_C_SVC
        params['kernel_type'] = cv2.ml.SVM_RBF
        return params


    def gen_mlp_params():
        params = {}
        params['train_method'] = cv2.ml.ANN_MLP_BACKPROP
        params['activation_func'] = cv2.ml.ANN_MLP_SIGMOID_SYM
        params['num_layers'] = 2
        params['num_layer_units'] = 50
        params['moment_scale'] = 0.1
        params['weight_scale'] = 0.1
        return params

    train_error_rates = [[] for model in models]
    test_error_rates = [[] for model in models]
    train_confusion_matrices = [0 for model in models]
    test_confusion_matrices = [0 for model in models]

    # Partition the training data in the set size
    train, train_labels, map = utils.partition(data, labels, True, class_size)

    # Our running accuracies for each K-fold
    avg_train_accuracies = [0 for model in models]
    avg_test_accuracies = [0 for model in models]

    # Auto train params
    params = []
    for model in models:
        if isinstance(model, svm.SVM):
            params.append(gen_svm_params())
        if isinstance(model, mlp.MLP):
            params.append(gen_mlp_params())

    for traincv, testcv in sklearn.cross_validation.KFold(len(train), n_folds=k_fold):
        for index, model in enumerate(models):
            # Train and test classifier
            model.auto_train(train[traincv], train_labels[traincv], params[index], k_folds=k_fold)
            train_predict = model.predict(train[traincv])
            test_predict = model.predict(train[testcv])

            # Calculate accuracy
            train_accuracy = 1.0*sum([1 for label, predict in zip(train_labels[traincv], train_predict) if label == predict]) / len(train_predict)
            test_accuracy = 1.0*sum([1 for label, predict in zip(train_labels[testcv], test_predict) if label == predict]) / len(test_predict)

            # Add to running average
            avg_train_accuracies[index] += train_accuracy
            avg_test_accuracies[index] += test_accuracy

            train_cm = sklearn.metrics.confusion_matrix(train_labels[traincv], train_predict, labels=np.arange(len(map)))
            test_cm = sklearn.metrics.confusion_matrix(train_labels[testcv], test_predict, labels=np.arange(len(map)))

            train_confusion_matrices[index] = train_cm
            test_confusion_matrices[index] = test_cm

    # Calculate average accuracy
    avg_train_accuracies = [accuracy/k_fold for accuracy in avg_train_accuracies]
    avg_test_accuracies = [accuracy/k_fold for accuracy in avg_test_accuracies]

    # Print average accuracy
    for index, model in enumerate(models):
        print '\n\tTraining accuracy for ' + model.__class__.__name__ + ': ', avg_train_accuracies[index]
        print '\tTesting accuracy for ' + model.__class__.__name__ + ': ', avg_test_accuracies[index]

    # Calculate error rates
    for index, model in enumerate(models):
        train_error_rates[index].append(1-avg_train_accuracies[index])
        test_error_rates[index].append(1-avg_test_accuracies[index])

    # Show confusion matrices and learning curves
#    for index, model in enumerate(models):
#        print '\n\tTraining confusion matrix for ' + model.__class__.__name__
#        show_confusion_matrix(train_confusion_matrices[index])
#        print '\n\tTesting confusion matrix for ' + model.__class__.__name__
#        show_confusion_matrix(test_confusion_matrices[index])
#        show_learning_curves(train_error_rates[index], test_error_rates[index], set_sizes)

    # Train and save models
    for index, model in enumerate(models):
        # Train the model
        model.auto_train(train, train_labels, params[index], k_folds=10)
     
        # Save the model
        now = datetime.datetime.now()
        name = now.strftime("%Y%m%d")[2:] + '_' + model.__class__.__name__ + '.xml'
        model.save(DEFAULT_OUTPUT_PATH+name)

    return models


def train(train_data, train_labels, dev_data, dev_labels, models):
    """ 
        This function trains some models and tests them using a development set

        Parameters
        ----------
        train_data:
        train_labels:
        dev_data:
        dev_labels:
        models:
            the models to train
        class_size:
            the amount of examples to use per class
    """
    pass 


def show_confusion_matrix(cm):
    """ 
        This function graphically displays a confusion matrix

        Parameters
        ----------
        cm: 2d array
            the confusion matrix to be displayed
    """
    norm_cm = []
    for i in cm:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/(float(a)+1e-6))
        norm_cm.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_cm), cmap=plt.cm.jet, interpolation='nearest')

    width = len(cm)
    height = len(cm[0])

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(cm[x][y]), xy=(y, x), horizontalalignment='center',verticalalignment='center')

    cb = fig.colorbar(res)
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    plt.xticks(range(width), alphabet[:width])
    plt.yticks(range(height), alphabet[:height])
    plt.show()


def show_learning_curves(train_error, test_error, data_sizes):
    """ 
        This functions plots the learning curves of a model.

        Parameters
        ----------
        train_error: list of floats
            list of training error rates
        test_error: list of floats
            list of testing error rates
        data_sizes: list of ints
            list of data sizes that corresponds to the training and testing error rates
            
    """
    plt.plot(data_sizes, train_error, data_sizes, test_error)
    plt.ylim([0, 1.0])
    plt.xlabel('Training Set Size')
    plt.ylabel('Error Rate')
    plt.show()


if __name__ == '__main__':
    main()
