class FeatureExtractor():
    """ 
        This is the base class for all feature extractors.
    """
    def extract(self, label, image):
        """ 
            Base function for feature extraction. All subclasses
            should implement this function.

            Parameters
            ----------
            label: str
                the corresponding label of the image
            image: numpy 3D (RGB) array
                the image to perform feature extraction on
        """
        raise NotImplementedError


    def extract_all(self, labels, images):
        """ 
            This function perfroms feature extraction on a list
            of labels and corresponding images. The default implementation
            simply applies the singular 'extract' function to each individual
            label and image pair. This function can be overridden if necessary.

            Parameters
            ----------
            labels: list of str
                the corresponding labels of each image
            images: list of numpy 3D (RGB) arrays
                the list of images to perform feature extraction on
        """
        features = []
        for label, image in zip(labels, images):
            feature = self.extract(label, image)
            features.append(feature)
        return features
