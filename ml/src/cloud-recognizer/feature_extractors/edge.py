import cv2
import numpy as np
from feature_extractor import FeatureExtractor


class EdgeExtractor(FeatureExtractor):
    """ 
        This class performs edge feature extraction.
    """
    def extract(self, label, image):
        """ 
            This function performs edge detection on the image and than returns
            a HoG feature descriptor for the image.

            Parameters
            ----------
            label: int
                the truth label of the image
            image: numpy 3D array (RGB)
                the image to perform the feature extraction on
        """
        edges = cv2.Canny(image,0,255)
        hog = cv2.HOGDescriptor()
        desc = hog.compute(edges).flatten()
        return desc

