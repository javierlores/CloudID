import cloud_classifier
import cv2
import numpy as np
import sklearn.cross_validation


DEFAULT_K_FOLDS = 10


class SVM(cloud_classifier.StatModel):
    """ 
        This class is an implementation of a support vector machine (SVM)
    """
    def __init__(self):
        self.model = cv2.ml.SVM_create()


    def load(self, filename):
        """ 
            This functions loads an SVM model from an .xml file

            Parameters
            ----------
            filename: path
                the file from which to load the model
        """
        self.model = cv2.ml.SVM_load(filename)


    def train(self, samples, responses, params):
        """ 
            This function trains the model with the parameters passed in.

            Parameters
            ----------
            samples: numpy array
                The data for which to train the neural network on
            responses: numpy array
                The labels for the corresponding samples
        """
        self.model.setKernel(params['kernel_type'])
        self.model.setType(params['svm_type'])
        self.model.setC(params['C'])
        self.model.setGamma(params['gamma'])
        self.model.setP(params['p'])
        self.model.setNu(params['nu'])
        self.model.setCoef0(params['coef'])
        self.model.setDegree(params['degree'])
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)


    def auto_train(self, samples, responses, params, k_folds=DEFAULT_K_FOLDS):
        """ 
            This function trains the model and selects the hyperparameters based on 
            cross-validation on the training set.

            Parameters
            ----------
            samples: numpy array
                The data for which to train the neural network on
            responses: numpy array
                The labels for the corresponding samples
            params: python dict
                A dictionary containing the following key/value pairs
                    svm_type: int 
                        SVM parameter
                    kernel_type: int 
                        SVM parameter
                    C: float
                        SVM parameter
                    gamma: float
                        SVM parameter
                    p: float
                        SVM parameter
                    nu: float
                        SVM parameter
                    coef: float
                        SVM parameter
                    degree: float
                        SVM parameter
        """

        def get_default_grid(id):
            """ 
                Returns a grid (python dict) containing the following key/value pairs
                    min_val: float
                        the minimum or starting value for the hyperparameter
                    max_val: float
                        the maximum or ending vlaue for the hyperparameter
                    step: int
                        the step size for the hyperparameter

                This grid is used to optimize the hyperparamters based on the previously stated values.

                Parameters
                ----------
                id: int
                    the id of the hyperparameter whose grid should be retrieved
            """
            grid = {}
            if id == cv2.ml.SVM_C:            # C 
                grid['min_val'] = 0.1
                grid['max_val'] = 1000
                grid['step'] = 2
            elif id == cv2.ml.SVM_GAMMA:      # Gamma
                grid['min_val'] = 0.00001
                grid['max_val'] = 2
                grid['step'] = 5
            elif id == cv2.ml.SVM_P:          # P
                grid['min_val'] = 0.001
                grid['max_val'] = 500
                grid['step'] = 3
            elif id == cv2.ml.SVM_NU:         # Nu
                grid['min_val'] = 0.001
                grid['max_val'] = 0.9
                grid['step'] = 2
            elif id == cv2.ml.SVM_COEF:       # Coef
                grid['min_val'] = 0.01
                grid['max_val'] = 500
                grid['step'] = 7
            elif id == cv2.ml.SVM_DEGREE:     # Degree
                grid['min_val'] = 0.001
                grid['max_val'] = 7
                grid['step'] = 3
            else:
                print 'Error, invalid grid_id'
                exit(1)

            return grid

        # Hyperparameters to optimize
        C = 0.0
        gamma = 0.0
        p = 0.0
        nu = 0.0
        coef = 0.0
        degree = 0.0

        # Optimized hyperparamter values
        best_C = 0.0
        best_gamma = 0.0
        best_p = 0.0
        best_nu = 0.0
        best_coef = 0.0
        best_degree = 0.0

        # The error to minimize
        min_error = float('inf')

        # Ensure sufficient cross-validation size
        if k_folds < 2:
            print 'K_fold value must be >= 2'
            exit(1)

        # Get default grids
        C_grid = get_default_grid(cv2.ml.SVM_C)
        gamma_grid = get_default_grid(cv2.ml.SVM_GAMMA)
        p_grid = get_default_grid(cv2.ml.SVM_P)
        nu_grid = get_default_grid(cv2.ml.SVM_NU)
        coef_grid = get_default_grid(cv2.ml.SVM_COEF)
        degree_grid = get_default_grid(cv2.ml.SVM_DEGREE)

        # These parameters are not used
        if params['svm_type'] == cv2.ml.SVM_NU_SVC or params['svm_type'] == cv2.ml.SVM_ONE_CLASS:
            C_grid['min_val'] = C_grid['max_val'] = params['C']
        if params['kernel_type'] == cv2.ml.SVM_LINEAR:
            gamma_grid['min_val'] = gamma_grid['max_val'] = params['gamma']
        if params['svm_type'] != cv2.ml.SVM_EPS_SVR:
            p_grid['min_val'] = p_grid['max_val'] = params['p']
        if params['svm_type'] == cv2.ml.SVM_C_SVC or params['svm_type'] == cv2.ml.SVM_EPS_SVR:
            nu_grid['min_val'] = nu_grid['max_val'] = params['nu']
        if params['kernel_type'] != cv2.ml.SVM_POLY and params['kernel_type'] != cv2.ml.SVM_SIGMOID:
            coef_grid['min_val'] = coef_grid['max_val'] = params['coef']
        if params['kernel_type'] != cv2.ml.SVM_POLY:
            degree_grid['min_val'] = degree_grid['max_val'] = params['degree']

        # Optimize hyper parameters
        C = C_grid['min_val']
        while True:
            gamma = gamma_grid['min_val']
            while True:
                p = p_grid['min_val']
                while True:
                    nu = nu_grid['min_val']
                    while True:
                        coef = coef_grid['min_val']
                        while True:
                            degree = degree_grid['min_val']
                            while True:
                                error = 0
                                for traincv, testcv in sklearn.cross_validation.KFold(len(samples), n_folds=k_folds):
                                    # Set model hyper parameters
                                    self.model.setType(params['svm_type'])
                                    self.model.setKernel(params['kernel_type'])
                                    self.model.setC(C)
                                    self.model.setGamma(gamma)
                                    self.model.setP(p)
                                    self.model.setNu(nu)
                                    self.model.setCoef0(coef)
                                    self.model.setDegree(degree)

                                    # Train the model
                                    self.model.train(samples[traincv], cv2.ml.ROW_SAMPLE, responses[traincv].astype('int32'))

                                    # Test the model
                                    ret, test_predict = self.model.predict(samples[testcv])
 
                                    # Sum error for current k_fold
                                    error += 1-1.0*sum([1 for label, predict in zip(responses[testcv], test_predict.ravel().astype('int32')) if label == predict]) / len(test_predict)

                                # Calculate the total error
                                error /= k_folds                                    

                                # Check if these hyper parameters give us the smallest error
                                if min_error > error:
                                    min_error = error
                                    best_degree = degree
                                    best_gamma = gamma
                                    best_coef = coef
                                    best_C = C
                                    best_nu = nu
                                    best_p = p

                                # Update
                                degree *= degree_grid['step']
                                # Check exit condition
                                if degree >= degree_grid['max_val']:
                                    break

                            # Update
                            coef *= coef_grid['step']
                            # Check exit condition
                            if coef >= coef_grid['max_val']:
                                break

                        # Update
                        nu *= nu_grid['step']
                        # Check exit condition
                        if nu >= nu_grid['max_val']:
                            break

                    # Update
                    p *= p_grid['step']
                    # Check exit condition
                    if p >= p_grid['max_val']:
                        break

                # Update
                gamma *= gamma_grid['step']
                # Check exit condition
                if gamma >= gamma_grid['max_val']:
                    break

            # Update
            C *= C_grid['step']
            # Check exit condition
            if C >= C_grid['max_val']:
                break

        self.model.setKernel(params['kernel_type'])
        self.model.setType(params['svm_type'])
        self.model.setC(best_C)
        self.model.setGamma(best_gamma)
        self.model.setP(best_p)
        self.model.setNu(best_nu)
        self.model.setCoef0(best_coef)
        self.model.setDegree(best_degree)
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)


    def predict(self, samples):
        """ 
            This functions predicts a label for each of the feature vectors passed in

            Parameters
            ----------
            samples: numpy array
                an array of feature vectors that will be used to predict a label
        """
        ret, results = self.model.predict(samples)
        return results
