import cv2
import numpy as np
from feature_extractor import FeatureExtractor


class AbsFourierTransformExtractor(FeatureExtractor):
    """ 
        This feature extraction extracts the absolute value of the fourier transform of the image
    """
    def extract(self, label, image):
        """ 
            This function extracts the absolute FFT of the image.

            Parameters
            ----------
            label: int
                the truth label of the image
            image: numpy 3D array (RGB)
                the image to perform the feature extraction on
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Compute the FFT of the gray image
        fft = np.fft.fft2(gray_image)
        fft_shift = np.fft.fftshift(fft)
        M = np.abs(np.log(fft_shift + 0.00001))

        return M.flatten()

