import csv
import random
import numpy as np

# Set how numpy handles errors
np.seterr(all='ignore')

DEFAULT_CLASS_MAP = '../../data/cloud-types.csv'


def read_features(feature_files):
    """ 
        This function reads all of the features in a list of feature files.
        Feature files are expected to be a csv file where each row corresponds
        to the features for one example.

        Parameters
        ----------
        features_files: list of file names
            the file names from which to read the featres
    """
    all_features = []
    all_labels = []
    for file_name in feature_files:
        with open(file_name, 'r') as csvfile:        
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')

            for index, row in enumerate(reader):
                # If this is the first time adding features, create an empty feature vector
                if not all_features:
                    all_features = [[] for i in range(int(row[0]))]
                    all_labels = [0 for i in range(int(row[0]))]
                # Otherwise, append the features to the appropriate examples
                else:
                    all_features[index-1].extend(row[:-1])
                    all_labels[index-1] = row[-1]

    return all_features, all_labels


def partition(data, labels, limit=False, size=10):
    """ 
        This function partitions the data and labels into balanced classes.

        Parameters
        ----------
        data: list of numpy arrays
            the data to partition
        labels: list of str
            corresponding data labels
        limit: boolean
            whether or not to limit the number of datums per class
        size: int
            the limit of the number of datums per class
    """
    def read_map():
        label_map = []
        label_counts = {}
        with open(DEFAULT_CLASS_MAP, 'r') as csvfile:        
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')

            for clazz in reader:
                label_map.append(clazz[0])
                label_counts[clazz[0]] = 0

        return label_map, label_counts

    # This will be used to store the images and their corresponding labels
    new_data = []
    new_labels = []

    # This will be used to store the label counts for each limit
    # It will be used to ensure a balanced dataset
    label_map, label_counts = read_map()

    # Randomly shuffle data
    indicies = random.sample(range(0, len(data)), len(data))

    # Select new data
    for index in indicies:
        datum = data[index]
        label = labels[index]

        # Increase the count of that label and adds the label and image to the new dataset
        if limit:
            if label_counts[label] < size:
                # Add item to new data
                new_labels.append(label_map.index(label))
                new_data.append(datum)

                # Increase class count
                label_counts[label] += 1
        else:
            new_labels.append(label_map.index(label))
            new_data.append(datum)

    return np.array(new_data, dtype='float32'), np.array(new_labels), label_map
