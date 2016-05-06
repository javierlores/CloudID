import numpy as np


class StatModel(object):
    """ 
        A base class for the machine learning models
    """
    def __init__(self, train_ratio=0.5, num_classes=10):
        self.train_ratio = train_ratio
        self.num_classes = num_classes


    def save(self, filename):
        """ 
            This functions saves a model from to an .xml file

            Parameters
            ----------
            filename: path
                the file to which to save the model
        """
        self.model.save(filename)


    def load(self, filename):
        """ 
            This functions loads a model from to an .xml file

            Parameters
            ----------
            filename: path
                the file to which to load the model
        """
        raise NotImplementedError, "Subclasses must implement load"


    def unroll_samples(self, samples):
        """ 

            Parameters
            ----------
            samples:
        """
        num_samples, num_features = samples.shape
        new_samples = np.zeros((num_samples * self.num_classes, num_features+1), np.float32)
        new_samples[:,:-1] = np.repeat(samples, self.num_classes, axis=0)
        new_samples[:,-1] = np.tile(np.arange(self.num_classes), num_samples)
        return new_samples


    def unroll_responses(self, responses):
        """ 

            Parameters
            ----------
            responses:
        """
        num_samples = len(responses)
        new_responses = np.zeros(num_samples*self.num_classes, np.int32)
        resp_idx = np.int32( responses + np.arange(num_samples)*self.num_classes )
        new_responses[resp_idx] = 1
        return new_responses


    def train(self, samples, responses):
        """ 
            This function trains the model with the parameters passed in.

            Parameters
            ----------
            samples: numpy array
                The data for which to train the neural network on
            responses: numpy array
                The labels for the corresponding samples
        """
        raise NotImplementedError, "Subclasses must implement train"
        

    def auto_train(self, samples, responses, params):
        """ 
            This function trains the model and selects the hyperparameters based on 
            cross-validation on the training set.

            Parameters:
            ----------
            samples: numpy array
                The data for which to train the neural network on
            responses: numpy array
                The labels for the corresponding samples
            params: dict
                Any necessary parameters to be passed in
        """
        raise NotImplementedError, "Subclasses must implement train"
        

    def predict(self, samples):
        """ 
            Perform the prediction of some samples using the trained model.

            Parameters
            ----------
            samples: numpy array
                The data samples to be predicted
        """
        raise NotImplementedError, "Subclasses must implement predict"
