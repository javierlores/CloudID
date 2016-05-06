import cv2
import numpy as np
from feature_extractor import FeatureExtractor


class SpectralExtractor(FeatureExtractor):
    """ 
       This class is a spectral feature extractor. 
    """
    def _kmeans_segmentation(self, image, steps=50, k=2):
        """ 
            This function performs k-means segmentation.

            Parameters
            ----------
            image: numpy 3D array (RGB)
                the image upon which to perform k-means segmentation
            steps: int
                the number of regions to divide the image into to increase computation speed
            k: int
                the number of clusters to segment the image into
        """
        steps = 50 # Image is divided in steps*steps region

        dx = image.shape[0] / steps
        dy = image.shape[1] / steps

        # Compute color features for each region
        features = []
        for x in range(steps):
            for y in range(steps):
                R = scipy.cluster.vq.mean(image[x*dx:(x+1)*dx, y*dy:(y+1)*dy, 0])
                G = scipy.cluster.vq.mean(image[x*dx:(x+1)*dx, y*dy:(y+1)*dy, 1])
                B = scipy.cluster.vq.mean(image[x*dx:(x+1)*dx, y*dy:(y+1)*dy, 2])
                features.append([R, G, B])
        features = np.array(features, 'f') # Make into array
        # Cluster
        centroids, variance = scipy.cluster.vq.kmeans(features, 2)
        code, distance = scipy.cluster.vq.vq(features, centroids)

        # Create image with cluster labels
        code_image = code.reshape(steps, steps)
        code_image = scipy.misc.imresize(code_image, image.shape[:2], interp='nearest')

        return code_image


    def _segment_image(self, image):
        """ 
            This function segments an image into cloud pixels and non-cloud pixels

            Parameters
            ----------
            image:
                the image upon which to perform k-means segmentation
        """
        # Segment the image and select the clusters
        segmented_image = self._kmeans_segmentation(image)
        cluster1 = image[np.where(segmented_image==0)]
        cluster2 = image[np.where(segmented_image==255)]
        # The cluster with the higher mean will be the cloud pixels since 
        # the clouds pixels will be lighter than the background
        if cluster1.mean() > cluster2.mean():
            cloud_pixels = cluster1
            non_cloud_pixels = cluster2
        else:
            cloud_pixels = cluster2
            non_cloud_pixels = cluster1

        return cloud_pixels, non_cloud_pixels


    def extract(self, label, image):
        """ 
            This function first segments the image into cloud pixels and non-cloud pixels. 
            Then it extracts the following 5 spectral features from the cloud pixels:
                1. Mean of Red channel
                2. Standard deviation of blue channel
                3. Difference between red and green channels
                4. Difference between red and blue channels
                5. Difference between blue and green channels
            Parameters
            ----------
            label: int
                the truth label of the image
            image: numpy 3D array (RGB)
                the image to perform the feature extraction on
        """
        features = []

        # Segment the image into cloud pixels and non-cloud pixels
        cloud_pixels, non_cloud_pixels = self._segment_image(image)

        # Mean of red channel
        red_mean = np.mean(cloud_pixels[..., 0])
        features.append(red_mean)

        # Standard deviation of blue channel
        blue_variance = np.var(cloud_pixels[..., 2])
        features.append(blue_variance)

        # Difference Between red and green channels
        red_green_difference = np.abs(np.mean(cloud_pixels[..., 0])-np.mean(cloud_pixels[..., 1]))
        features.append(red_green_difference)

        # Difference Between red and blue channels
        red_blue_difference = np.abs(np.mean(cloud_pixels[..., 0])-np.mean(cloud_pixels[..., 2]))
        features.append(red_blue_difference)

        # Difference Between blue and green channels
        blue_green_difference = np.abs(np.mean(cloud_pixels[..., 2])-np.mean(cloud_pixels[..., 1]))
        features.append(blue_green_difference)

        return np.array(features)

