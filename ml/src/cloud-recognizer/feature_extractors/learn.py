
#! /usr/bin/env python2.7


import argparse
import cv2
import numpy as np
import skimage.feature
import scipy.misc
import scipy.ndimage
import csv
import os
import datetime

from bow import BagOfVisualWordsExtractor
from lbp import LBPExtractor
from spectral import SpectralExtractor
from textural import TexturalExtractor
from edge import EdgeExtractor
from fft import AbsFourierTransformExtractor


DEFAULT_DATA_PATH = '../../../data/'
DEFAULT_IMAGE_SIZE = (20, 20)
DEFAULT_SET = 'train'
	

def main():
    # Extract and assign arguments
    args = get_arguments()
    
    # The parameters of extraction
    image_size = (args.image_size, args.image_size) if args.image_size is not None else DEFAULT_IMAGE_SIZE
    data_path = args.data_path if args.data_path is not None else DEFAULT_DATA_PATH
    set = args.set if args.set is not None else DEFAULT_SET

    # The features to extract
    extract_bow = args.bow
    extract_lbp = args.lbp
    extract_spec = args.spec
    extract_text = args.text
    extract_edge = args.edge
    extract_abs = args.abs

    # Read in the dataset
    images, labels = read_dataset(data_path, set, image_size)

    # Create feature extractors
    feature_extractors = []

    if extract_bow: feature_extractors.append(BagOfVisualWordsExtractor())
    if extract_lbp: feature_extractors.append(LBPExtractor())
    if extract_spec: feature_extractors.append(SpectralExtractor())
    if extract_text: feature_extractors.append(TexturalExtractor())
    if extract_edge: feature_extractors.append(EdgeExtractor())
    if extract_abs: feature_extractors.append(AbsFourierTransformExtractor())

    # Extract and write features to file
    # Name the files as yymmdd_<feature_name>_<img_size>_<set>
    for feature_extractor in feature_extractors:
        # Extract the features
        features = feature_extractor.extract_all(labels, images)

        # Create output file name
        now = datetime.datetime.now()
        file_name = data_path+'features/'+now.strftime("%Y%m%d")[2:] + '_' + feature_extractor.__class__.__name__ + '_' + str(image_size).replace(" ", "") + '_' + set + '.csv'
        write_to_file(file_name, features, labels)


def get_arguments():
    """ 
        This function retrieves the command line parameters passed into this script.

        Parameters
        ----------
        None
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, help='Specify the path to the dataset')
    parser.add_argument('--set', type=str, help='Specify the set from which to extract features')
    parser.add_argument('--image-size', type=int, help='Specify the size of each image')
    parser.add_argument('--bow', action='store_true', help='Specify to enable BoW feature extraction')
    parser.add_argument('--lbp', action='store_true', help='Specify to enable LBP feature extraction')
    parser.add_argument('--spec', action='store_true',  help='Specify to enable spectral feature extraction')
    parser.add_argument('--text', action='store_true', help='Specify to enable textural feature extraction')
    parser.add_argument('--edge', action='store_true', help='Specify to enable edge feature extraction')
    parser.add_argument('--abs', action='store_true', help='Specify to enable abs fourier transform feature extraction')
    args = parser.parse_args()

    return args


def read_dataset(data_path, set, image_size):
    """ 
        This function reads the images and correspondings labels of the dataset.
        data_path is assumed to contain a subdirectory called 'sets' that contains
        text files for each set and class (e.g. <class>_<set>.txt) that specify which 
        images (located in the subdirectory called 'images') belong to that set and class.

        Parameters
        ----------
        data_path : str
            the path to the location of the images in the dataset
        set: (train|train_aug|dev|dev_aug|test)
            the set from which to extract the features
        image_size : int
            the dimensions of the image
    """
    # This will be used to store the images and their corresponding labels
    labels = []
    images = []

    # Read in images for the the specified set
    for subdir, dirs, files in os.walk(data_path+'sets/'):
        for file in files:
            # Check if the file belongs to the desired set
            if set == file.split('_')[-1][:-4]:
                label = file.split('_')[0]

                # Read images up to class_size
                with open(os.path.join(data_path+'sets/', file), 'r') as output_file:
                    for image_name in output_file:
                        # Read in the image as color
                        image = cv2.imread(os.path.join(data_path+'images/', image_name.rstrip('\n')), 1)

                        # If everything was successful, resize the image and add it
                        if image is not None:
                            # Add the image and its' label to the dataset
                            images.append(image)
                            labels.append(label)
    return images, labels


def write_to_file(file_name, data, labels):
    """ 
        This function writes features learned and corresponding labels to a .csv file

        Parameters
        ----------
        file_name : str
            The file name to save the features
        data : numpy array
            The features to be saved
        labels : numpy array
            The corresponding labels for the features
    """
    with open(file_name, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)    

        writer.writerow([len(labels)])
        for datum, label in zip(data, labels):
            writer.writerow(datum.tolist() + [label])


if __name__ == '__main__':
    main()
