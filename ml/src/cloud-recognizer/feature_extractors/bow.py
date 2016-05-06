import cv2
import numpy as np
import scipy.cluster.vq

from feature_extractor import FeatureExtractor


class BagOfVisualWordsExtractor(FeatureExtractor):
    """ 
        This class is the bag of visual words extractor.
        SIFT is used for the feature descriptor.
    """
    def extract(self, label, image, method='SIFT'):
        """ 
            This function extracts the SIFT features for a single image.

            Parameters
            ----------
            label: int
                the truth label of the image
            image: numpy 3D array (RGB)
                the image from which to extract the SIFT features.
            method: (SIFT|SURF)
                the method to use for the feature descriptor
        """
        # Convert the image to gray scale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Extract SIFT features
        if method == 'SIFT':
            sift = cv2.xfeatures2d.SIFT_create()
            kps, desc = sift.detectAndCompute(gray_image, None)
        # Extract SURF features
        if method =='SURF':
            surf = cv2.xfeatures2d.SURF_create()
            kps, desc = surf.detectAndCompute(gray_image, None)

        return desc


    def extract_all(self, labels, images, method='SIFT', k_thresh=1, pre_alloc_buff=1000):
        """ 
            This function performs BoW extraction on a list of images and labels

            Parameters
            ----------
            labels: list of ints
                the truth labels of each image
            images: list of numpy 3D (RGB) arrays
                the images from which features will be extracted
            method: (SIFT|SURF)
                the method to use for the feature descriptor
        """
        if method == 'SIFT': self.feat_len = 128
        if method == 'SURF': self.feat_len = 64

        # Create a dictionary to store the features
        features = {}
        counter = 1
        for label, image in zip(labels, images):
            desc = self.extract(label, image)
            if desc is not None:
                features['image-'+str(counter)] = desc
                counter += 1
            else:
                features['image-'+str(counter)] = np.zeros((1, self.feat_len))
                counter += 1

        # Convert dictionary to numpy array
        # This step is necessary because the descriptors must be
        # Of the same dimensions
        all_features = self._dict_to_numpy(features, pre_alloc_buff)

        # Computing visual words with k-means
        k = int(np.sqrt(all_features.shape[0]))
        codebook, variance = scipy.cluster.vq.kmeans(all_features, k, thresh=k_thresh)

        # Compute histograms
        histograms = []
        for index, feature in enumerate(features):
            code, distance = scipy.cluster.vq.vq(features['image-'+str(index+1)], codebook)
            histogram, bin_edges = np.histogram(code, bins=range(codebook.shape[0]+1), normed=True)
            histograms.append(histogram)

        return np.array(histograms)


    def _dict_to_numpy(self, dict, pre_alloc_buff):
        """ 
            This function converts a python dictionary to a numpy array

            Parameters
            ----------
            dict: python dictionary
                the dictionary to be converted
            pre_alloc_buffer: int
                the amount of space to preallocate in the numpy array
                for each key in the python dictionary
        """
        nkeys = len(dict)
        array = np.zeros((nkeys*pre_alloc_buff, self.feat_len))
        pivot = 0
        for key in dict.keys():
            value = dict[key]
            nelements = value.shape[0]
            while pivot + nelements > array.shape[0]:
                padding = zeros_like(array)
                array = np.vstack((array, padding))
            array[pivot:pivot + nelements] = value
            pivot += nelements
        array = np.resize(array, (pivot, self.feat_len))
        return array

