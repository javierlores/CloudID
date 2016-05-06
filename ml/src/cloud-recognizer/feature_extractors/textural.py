import cv2
import numpy as np
from feature_extractor import FeatureExtractor


class TexturalExtractor(FeatureExtractor):
    """ 
        This class creates the gray-level co-occurence matrix(GLCM) of an image
        and then extracts the following 12 textural features:
            1. Contrast
            2. Homogeneity
            3. Energy
            4. Variance
            5. Inverse Difference Moment
            6. Sum Average
            7. Sum Variance
            8. Sum Entropy
            9. Entropy
           10. Difference Variance
           11. Difference Entropy
           12. Correlation
    """
    def _calc_glcm(self, image, offset, rotation_offset):
        """ 
            Calculate gray-level co-occurence matrix (GLCM).

            Parameters:
                image:
                    the image to calculate the GLCM of
                offset:
                    the GLCM offset
                rotation_offset
                    the rotational GLCM offset
        """
        rows, cols = image.shape
        glcm = np.zeros((256,256))
        for y in range(offset[0], rows-offset[0]):
            for x in range(offset[1], cols-offset[1]):
                p = image[y][x]
                q = image[y+offset[0]][x+offset[1]]

                glcm[p][q] += 1

        return glcm


    def _calc_contrast(self, glcm):
        """ 
            This function calculates the contrast over the GLCM

            Parameters
            ----------
            glcm: 2D numpy array
                the GLCM from which to calculate the contrast
        """
        row_indices = np.zeros(glcm.shape)
        col_indices = np.zeros(glcm.shape)

        row_indices[:, :] = np.arange(256).reshape((256,1))
        col_indices[:, :] = np.arange(256)

        contrast = np.sum(glcm*(row_indices-col_indices)**2)

        return contrast


    def _calc_homogeneity(self, glcm):
        """ 
            This function calculates the homogeneity over the GLCM

            Parameters
            ----------
            glcm: 2D numpy array
                the GLCM from which to calculate the homogeneity
        """
        row_indices = np.zeros(glcm.shape)
        col_indices = np.zeros(glcm.shape)

        row_indices[:, :] = np.arange(256).reshape((256,1))
        col_indices[:, :] = np.arange(256)

        homogeneity = np.sum(glcm/(1+np.abs(row_indices-col_indices)))

        return homogeneity


    def _calc_energy(self, glcm):
        """ 
            This function calculates the energy over the GLCM

            Parameters
            ----------
            glcm: 2D numpy array
                the GLCM from which to calculate the energy
        """
        energy = np.sum(glcm**2)

        return energy


    def _calc_variance(self, glcm):
        """ 
            This function calculates the variance over the GLCM

            Parameters
            ----------
            glcm: 2D numpy array
                the GLCM from which to calculate the variance
        """
        row_indices = np.zeros(glcm.shape)
        col_indices = np.zeros(glcm.shape)

        row_indices[:, :] = np.arange(256).reshape((256,1))
        col_indices[:, :] = np.arange(256)

        variance = np.sum(glcm/(1+np.abs(row_indices-col_indices)))

        return variance


    def _calc_inverse_difference_moment(self, glcm):
        """ 
            This function calculates the inverse difference moment over the GLCM

            Parameters
            ----------
            glcm: 2D numpy array
                the GLCM from which to calculate the inverse difference moment
        """
        row_indices = np.zeros(glcm.shape)
        col_indices = np.zeros(glcm.shape)

        row_indices[:, :] = np.arange(256).reshape((256,1))
        col_indices[:, :] = np.arange(256)

        inverse_difference_moment = np.sum(glcm/(1+(row_indices-col_indices)**2))

        return inverse_difference_moment


    def _calc_sum_average(self, glcm):
        """ 
            This function calculates the sum average over the GLCM

            Parameters
            ----------
            glcm: 2D numpy array
                the GLCM from which to calculate the sum average
        """
        glcm_indices = np.sum(np.indices(glcm.shape), axis=0).flatten()
        glcm_flat = glcm.flatten()
#        glcm_sum = ((np.mgrid[:glcm_indices.max(), :glcm_flat.shape[0]] == glcm_indices)[0] * glcm_flat), :)

        sum_indices = np.arange(glcm_indices.max())
        sum_average = np.sum(sum_indices*glcm_sum)

        return sum_average


    def _calc_sum_variance(self, glcm):
        """ 
            This function calculates the sum variance over the GLCM

            Parameters
            ----------
            glcm: 2D numpy array
                the GLCM from which to calculate the sum variance
        """
        glcm_indices = np.sum(np.indices(glcm.shape), axis=0).flatten()
        glcm_flat = glcm.flatten()
#        glcm_sum = ((np.mgrid[:glcm_indices.max(), :glcm_flat.shape[0]] == glcm_indices)[0] * glcm_flat)$

        sum_entropy = self._calc_sum_entropy(glcm)

        sum_indices = np.arange(glcm_indices.max())
        sum_variance = np.sum(glcm_sum*(sum_indices-sum_entropy)**2)

        return sum_variance


    def _calc_sum_entropy(self, glcm):
        """ 
            This function calculates the sum entropy over the GLCM

            Parameters
            ----------
            glcm: 2D numpy array
                the GLCM from which to calculate the sum entropy
        """
        glcm_indices = np.sum(np.indices(glcm.shape), axis=0).flatten()
        glcm_flat = glcm.flatten()
#        glcm_sum = ((np.mgrid[:glcm_indices.max(), :glcm_flat.shape[0]] == glcm_indices)[0] * glcm_flat)

        sum_entropy = np.sum(glcm_sum*np.log(glcm_sum+0.000001))

        return sum_entropy


    def _calc_entropy(self, glcm):
        """ 
            This function calculates the entropy over the GLCM

            Parameters
            ----------
            glcm: 2D numpy array
                the GLCM from which to calculate the entropy
        """
        entropy = -np.sum(glcm*np.log(glcm+0.000001))

        return entropy


    def _calc_difference_variance(self, glcm):
        """ 
            This function calculates the difference variance over the GLCM

            Parameters
            ----------
            glcm: 2D numpy array
                the GLCM from which to calculate the difference variance
        """
        glcm_indices = np.sum(np.indices(glcm.shape), axis=0).flatten()
        glcm_flat = glcm.flatten()
        glcm_sum = ((np.mgrid[:a.shape, :glcm_flat.shape] == glcm_indices)[0] * glcm_flat).sum(axis=1)

        sum_entropy = self._calc_sum_entropy(glcm)

        sum_indicies = np.arange(510)
        difference_variance = np.sum(glcm_sum*(sum_indices-sum_entropy)**2)

        return difference_variance


    def _calc_difference_entropy(self, glcm):
        """ 
            This function calculates the difference entropy of the GLCM

            Parameters
            ----------
            glcm: 2D numpy array
                the GLCM from which to calculate the difference entropy
        """
        glcm_indices = np.sum(np.indices(glcm.shape), axis=0).flatten()
        glcm_flat = glcm.flatten()
        glcm_sum = ((np.mgrid[:a.shape, :glcm_flat.shape] == glcm_indices)[0] * glcm_flat).sum(axis=1)

        sum_entropy = self._calc_sum_entropy(glcm)

        sum_indicies = np.arange(510)
        difference_entropy = np.sum(glcm_sum*(sum_indices-sum_entropy)**2)

        return difference_entropy


    def _calc_correlation(self, glcm):
        """ 
            This function calculates the correlation from the GLCM as

            
            Parameters
            ----------
            glcm: 2D numpy array
                the GLCM from which to calculate the correlation
        """
        a = np.zeros(glcm.shape)
        col_indices = np.zeros(glcm.shape)
        inverse_difference_moment = np.zeros(glcm.shape)

        a[:, :] = np.arange(256)
        col_indices[:, :] = np.arange(256).reshape((256,1))
        inverse_difference_moment = glcm/(1+(a-col_indices)**2)

        return inverse_difference_moment
 

    def extract(self, label, image):
        """ 
            This function extracts the haralick features of an image

            Parameters
            ----------
            label: int
                the truth label of the image
            image: numpy 3D array (RGB)
                the image to perform the feature extraction on
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate GLCM
        glcm = self._calc_glcm(gray_image, [1,1], [])

        # The functions used to calculate the haralick features
        haralick_feature_funcs = [self._calc_contrast, \
                                  self._calc_homogeneity, \
                                  self._calc_energy, \
                                  self._calc_variance, \
                                  self._calc_inverse_difference_moment, \
#                                  self._calc_sum_average, \
#                                  self._calc_sum_variance, \
#                                  self._calc_sum_entropy, \
                                  self._calc_entropy]
#                                  self._calc_difference_variance, \
#                                  self._calc_difference_entropy, \
#                                  self._calc_correlation]

        # Where we will store the textural features calculated
        features = []

        # Calculate each feature and add it to our feature vector
        for feature_func in haralick_feature_funcs:
            feature = feature_func(glcm)
            features.append(feature)

        return np.array(features)

