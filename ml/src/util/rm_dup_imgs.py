#! /usr/bin/env python2.7


import PythonMagick
import os
import argparse
import numpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


DEFAULT_DATA_PATH = '../../data/'


def main():
    """ 
        This function will iterate through a directory and check for duplicate images.
        When a possible duplicate image is found (Based on image hash), the 2 images
        will be displayed and the user will be asked if the duplicate should removed.

        Parameters
        ----------
        None
    """

    # Parse arguments
    args = parse_arguments()

    # Extract the arguments
    data_path = args.data_path if args.data_path is not None else DEFAULT_DATA_PATH

    # Read in the image
    img_dict = {}
    for subdir, dirs, files in os.walk(data_path):
        for file in files:
            if not file.endswith('.jpg'):
                continue
            # Open the img and get its hash
            img_loc = os.path.join(subdir, file)
            img_hash = PythonMagick.Image(img_loc).signature()

            # Check if the hash exists
            if img_hash in img_dict:
                # Show images
                img1 = mpimg.imread(img_loc)
                img2 = mpimg.imread(img_dict[img_hash])

                fig = plt.figure()
                a=fig.add_subplot(1,2,1)
                plt.imshow(img1)
                a.set_title('Original')
                a=fig.add_subplot(1,2,2)
                plt.imshow(img2)
                a.set_title('Duplicate')
                plt.show()

                # Ask user if ok to delete
                print img_loc + ' is a duplicate of ' + img_dict[img_hash]
                answer = raw_input('would you like to delete the duplicate (Y/N)?')

                # Delete duplicate
                if answer.lower() == 'y':
                    os.remove(img_loc)
                    print 'removed'
            else:
                img_dict[img_hash] = img_loc
            

def parse_arguments():
    """ 
        This function retrieves the command line parameters passed into this script.

        Parameters
        ----------
        None
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--data-path', type=str, help='Specify the path to the dataset')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
