import cv2
import numpy as np
import skimage.feature
from feature_extractor import FeatureExtractor


class LBPExtractor(FeatureExtractor):
    """ 
        This class Local Binary Patterns extractor.
    """
    def extract(self, label, image, method='nri_uniform', radius=4):
        """ 
            This function extracts the uniform and non rotationally invariant local
            binary patterns with radius of 3 of an image.

            Parameters
            ----------
            label: int
                the truth label of the image
            image: numpy 3D (RGB) array
                the image from which to extract features
            method: (nri_uniform|ror|uniform|var)
                the method to perform the LBP extraction
            radius: int
                the radius of the LBP neighborhood
        """
        # PREPROCESSING
        # Convert image to gray scale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Normalize the image
        gray_image = gray_image.astype('float32')/255.0

        # Extract LBP
        lbp = skimage.feature.local_binary_pattern(gray_image, 6*radius, radius, method)

        nbins = lbp.max()+1
        hist, _ = np.histogram(lbp, normed=True, bins=nbins, range=(0, nbins))
        return hist

