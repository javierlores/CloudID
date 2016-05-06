#! /usr/bin/env python2.7


import sys
import argparse
import os
import numpy as np 
import datetime

from shutil import copyfile

DEFAULT_IMAGE_PATH = '../../data/images/'
DEFAULT_SET_PATH = '../../data/sets/'


def main():
    """ 
        The main logic function.

        Parameters
        ----------
        None
    """
    # Parse arguments
    args = parse_arguments()

    # Ensure sufficient the arguments
    if args.data_path is None:
        print 'Error, need to specify data path'
        sys.exit(1)

    # Extract arguments
    data_path = args.data_path
    image_path = args.image_path if args.image_path is not None else DEFAULT_IMAGE_PATH
    set_path = args.set_path if args.set_path is not None else DEFAULT_SET_PATH

    # Create directories if they don't exist
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    if not os.path.exists(set_path):
        os.makedirs(set_path)

    # Read in the data
    data, labels = read_data(data_path)

    # Rename and save the data in the new image directory
    data = resave_files(data, image_path)

    # Split the data
    train_labels, dev_labels, test_labels, train_data, dev_data, test_data = split(data, labels, 0.8, 0, 0.2)

    # Write the data to files
    export(set_path, train_labels, dev_labels, test_labels, train_data, dev_data, test_data)


def parse_arguments():
    """ 
        This function retrieves the command line parameters passed into this script.

        Parameters
        ----------
        None
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--data-path', type=str, help='Specify the path to the images to be split')
    parser.add_argument('-ip', '--image-path', type=str, help='Specify the path to save the split files')
    parser.add_argument('-sp', '--set-path', type=str, help='Specify the path to save the split details txt files')
    args = parser.parse_args()

    return args


def read_data(data_path):
    """ 
        This function reads in the data and corresponding labels.
        The assumed format is that each datum resides in a directory
        whose names is also the label for that datum

        Parameters
        ----------
        data_path: str
            the path to the location of the directories containing the data
    """
    data = []
    labels = []
    for subdir, dirs, files in os.walk(data_path):
        label = subdir.split('/')[-1]   # The label for the datum is the name of the directory

        for file in files:
            data.append((subdir, file))
            labels.append(label)

    return np.array(data), np.array(labels)


def resave_files(data, image_path):
    """ 
        This function renames files in a directory. 
        The names of the files will be formatted as
        date-id in the format yymmdd-0001
        For example 010316-0000001

        Parameters
        ----------
        None
    """
    now = datetime.datetime.now()   # Current datetime for creating the new filenames
    counter = 1                     # A counter to create the new filenames
    new_data = []                   # Location to store new filenames

    # Read the files and rename, and copy them
    for subdir, file in data:
        # Create our new file name
        new_name = now.strftime("%Y%m%d")[2:] + '_' + str(counter).zfill(7) + '.' + file.split('.')[-1]

        # Copy the file to our new image directory
        copyfile(os.path.join(subdir, file), os.path.join(image_path, new_name))
        counter += 1               # Counter for file names
        new_data.append(new_name)  # Add the new file name

    return np.array(new_data)


def split(data, labels, train_per=0.8, dev_per=0.0, test_per=0.2):
    """ 
        This function splits the data and labels into the training, development/validation, 
        and test sets based on the percentages passed in.

        Parameters
        ----------
        data: numpy array
            the data to be split
        labels: numpy array
            the corresponding labels for the data
        train_per (optional): float
            the percentage of data to be allocated for the training set
        dev_per (optional):  float
            the percentage of data to be allocated for the development/validation set
        test_per (optional): float
            the percentage of data to be allocated for the test set
    """
    # Ensure proper percentages for each set are passed in
    if (train_per + dev_per + test_per) != 1:
        print "train, dev, and test splits should sum to one"
        return

    # Randomize data and labels
    new_order = np.arange(data.shape[0])
    np.random.seed(0)  # set seed
    np.random.shuffle(new_order)
    data = data[new_order]
    labels = labels[new_order]

    # If there is to be no development/validation set
    if dev_per == 0:
        dim = labels.shape[0]
        split1 = int(dim*train_per)            # The train/test boundary
        train_labels = labels[0:split1]          # Split training set labels
        test_labels = labels[split1:]            # Split test set labels
        dev_labels = np.array([])
        train_data = data[0:split1]            # Split training set data
        test_data = data[split1:]              # Split test set data
        dev_data = np.array([])
    # If there is a development/validation set
    else:
        dim = labels.shape[0]
        split1 = int(dim*train_per)            # The train/dev boundary
        split2 = int(dim*(train_per+dev_per))  # The dev/test boundary
        train_labels = labels[0:split1]          # Split training set labels
        dev_labels = labels[split1:split2]       # Split development set labels
        test_labels = labels[split2:]            # Split test set labels
        train_data = data[0:split1]            # Split training set data
        dev_data = data[split1:split2]         # Split development set data
        test_data = data[split2:]              # Split test set data

    return train_labels, dev_labels, test_labels, train_data, dev_data, test_data


def export(set_path, train_labels, dev_labels, test_labels, train_data, dev_data, test_data):
    """ 
        This function writes the training set, development set, and testing set to text files.
        The text files are named in the format <class>_<set>.txt where class is the label.

        Parameters
        ----------
        set_path: str
            the path to store the set files
        train_labels: numpy array
            the labels for the training set
        dev_labels: numpy array
            the labels for the dev set
        test_labels: numpy array
            the labels for the test set
        train_data: numpy array
            the data for the training set
        dev_data: numpy array
            the data for the dev set
        test_data: numpy array
            the data for the test set
    """
    # Export the train set
    for datum, label in zip(train_data, train_labels):
        file_name = set_path+label+'_train.txt'

        with open(file_name, 'a') as file:
            file.write(datum+'\n')

    # Export the dev set
    for datum, label in zip(dev_data, dev_labels):
        file_name = set_path+label+'_dev.txt'

        with open(file_name, 'a') as file:
            file.write(datum+'\n')

    # Export the test set
    for datum, label in zip(test_data, test_labels):
        file_name = set_path+label+'_test.txt'

        with open(file_name, 'a') as file:
            file.write(datum+'\n')


if __name__ == '__main__':
    main()
