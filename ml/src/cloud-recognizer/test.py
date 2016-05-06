#! /usr/bin/env python2.7

import sys 
sys.path.append("../util/") 
import csv 
import cv2 
import numpy as np 
import random 
import argparse 
import models.cloud_classifier 
import models.svm as svm 
import models.mlp as mlp 
import utils


DEFAULT_CLASS_SIZE = 100


def main():
    # Read the arguments
    args = parse_arguments()

    # Get the class size
    class_size = args.class_size if args.class_size is not None else DEFAULT_CLASS_SIZE

    # Read the features from the test files
    test_files = args.test_files

    # Ensure at least 1 test file is passed in
    if test_files is None:
        print 'Error. Please provide testing feature files'
        exit(1)

    test_data, test_labels = utils.read_features(test_files)
    test_data, test_labels, map = utils.partition(test_data, test_labels, class_size)

    # Read and load the model
    if args.svm:
        model = svm.SVM()
        model.load(args.model)
    if args.mlp:
        model = mlp.MLP(10)
        model.load(args.model)

    # Ensure a model was created
    if model is None:
        print 'Error. Model invalid'
        exit(1)

    # Test the model
    predictions = model.predict(test_data)
    accuracy = 1.0*sum([1 for label, predict in zip(test_labels, predictions) if label == predict]) / len(predictions)

    # Output results
    print 'Accuracy is: ', accuracy


def parse_arguments():
    """ 
        This function retrieves the command line parameters passed into this script.

        Parameters
        ----------
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-files', type=str, nargs='+', help='List the names of the testing feature files')
    parser.add_argument('--class-size', type=int, help='the number of images per class to use in the test set')
    parser.add_argument('--svm', action='store_true', help='Specify to enable SVM classifier')
    parser.add_argument('--mlp', action='store_true', help='Specify to enable MLP ANN classifier')
    parser.add_argument('--model', type=str, help='The model file')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
