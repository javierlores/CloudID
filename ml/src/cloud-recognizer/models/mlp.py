import cloud_classifier
import cv2
import numpy as np
import sklearn.cross_validation


DEFAULT_K_FOLD = 10


class MLP(cloud_classifier.StatModel):
    """ 
        This class is an implementation of a Multi-layer Perceptron (MLP) artificial neural network (ANN)
    """
    def __init__(self, num_classes):
        self.model = cv2.ml.ANN_MLP_create()
        self.num_classes = num_classes


    def load(self, filename):
        """ 
            This functions loads a MLP model from an .xml file

            Parameters
            ----------
            filename: path
                the file from which to load the model
        """
        self.model = cv2.ml.ANN_MLP_load(filename)


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
        num_samples, num_features = samples.shape
        new_responses = self.unroll_responses(responses).reshape(-1, self.num_classes)
        self.model.setLayerSizes(params['layer_sizes'])
        self.model.setActivationFunction(params['activation_func'])
        self.model.setBackpropMomentumScale(params['moment_scale'])
        self.model.setBackpropWeightScale(params['weight_scale'])
        self.model.setTrainMethod(params['train_method'])
        self.model.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 1000, 0.01))
        self.model.train(samples, cv2.ml.ROW_SAMPLE, np.float32(new_responses))


    def auto_train(self, samples, responses, params, k_folds=DEFAULT_K_FOLD):
        """ 
            This function trains the model and selects the hyperparameters based on 
            cross-validation on the training set.

            Parameters:
            ----------
            samples: numpy array
                The data for which to train the neural network on
            responses: numpy array
                The labels for the corresponding samples
            params:
                num_layers: int
                    The number of hidden layers
                num_layer_units: int
                    The number of neurons per hidden layer
                activation_func: int
                    The activation function
                train_method: int
                    Backprop or Rprop
                bp_moment_scale: float
                    BackProp parameter
                bp_weight_scale: float
                    BackProp parameter
                rp_dw0: float
                    RProp parameter
                rp_dw_plus: float
                    RProp parameter
                rp_dw_minus: float
                    RProp parameter
                rp_dw_min: float
                    RProp parameter
                rp_dw_max: float
                    RProp parameter
            k_fold: int
                the number of folds to use for cross-validation
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
            if id == 'NUM_LAYERS':
                grid['min_val'] = 1
                grid['max_val'] = 4
                grid['step'] = 2
            elif id == 'NUM_LAYER_UNITS':
                grid['min_val'] = 2
                grid['max_val'] = 300
                grid['step'] = 2
            elif id == 'BP_MOMENT':
                grid['min_val'] = 0.0001
                grid['max_val'] = 1
                grid['step'] = 5
            elif id == 'BP_DW':
                grid['min_val'] = 0.0001
                grid['max_val'] = 1
                grid['step'] = 5
            elif id == 'RP_DW0':
                grid['min_val'] = 0.0001
                grid['max_val'] = 1
                grid['step'] = 5
            elif id == 'RP_DW_PLUS':
                grid['min_val'] = 0.0001
                grid['max_val'] = 1
                grid['step'] = 5
            elif id == 'RP_DW_MINUS':
                grid['min_val'] = 0.0001
                grid['max_val'] = 1
                grid['step'] = 5
            elif id == 'RP_DW_MIN':
                grid['min_val'] = 0.0001
                grid['max_val'] = 1
                grid['step'] = 5
            elif id == 'RP_DW_MAX':
                grid['min_val'] = 0.0001
                grid['max_val'] = 1
                grid['step'] = 5
            return grid

        # Optimize for BackProp algorithm 
        if params['train_method'] == cv2.ml.ANN_MLP_BACKPROP:
            num_samples, num_features = samples.shape
            new_responses = self.unroll_responses(responses).reshape(-1, self.num_classes)

            # Hyperparameters to optimize
            num_layers = 0
            num_layer_units = 0
            bp_moment_scale = 0.0
            bp_dw_scale = 0.0

            # Optimized hyperparameter values
            best_num_layers = 0
            best_num_layer_units = 0
            best_bp_moment_scale = 0.0
            best_bp_dw_scale = 0.0

            # The error to minimize
            min_error = float('inf')

            # Ensure sufficient cross-validation size
            if k_folds < 2:
                print 'K_folds value must be >= 2'
                exit(1)

            # Get default grids
            num_layers_grid = get_default_grid('NUM_LAYERS')
            num_layer_units_grid = get_default_grid('NUM_LAYER_UNITS')
            moment_scale_grid = get_default_grid('BP_MOMENT')
            weight_scale_grid = get_default_grid('BP_DW')

            # Optimize hyper parameters
            num_layers = num_layers_grid['min_val']
            while True:
                num_layer_units = num_layer_units_grid['min_val']
                while True:
                    moment_scale = moment_scale_grid['min_val']
                    while True:
                        weight_scale = weight_scale_grid['min_val']
                        while True:
                            error = 0
                            for traincv, testcv in sklearn.cross_validation.KFold(len(samples), n_folds=k_folds):
                                # Set model hyper parameters
                                layer_sizes = np.int32([num_features]+[num_layer_units for i in range(num_layers)]+[self.num_classes])
                                self.model.setLayerSizes(layer_sizes)
                                self.model.setTrainMethod(params['train_method'])
                                self.model.setBackpropMomentumScale(moment_scale)
                                self.model.setBackpropWeightScale(weight_scale)
                                self.model.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 0.01))
                                self.model.setActivationFunction(params['activation_func'], 2, 1)

                                # Train the model
                                self.model.train(samples[traincv], cv2.ml.ROW_SAMPLE, np.float32(new_responses[traincv]))

                                # Test the model
                                ret, test_predict = self.model.predict(samples[testcv])

                                # Sum over the error
                                error += 1-1.0*sum([1 for label, predict in zip(new_responses[testcv].argmax(-1), test_predict.argmax(-1)) if label == predict]) / len(test_predict.argmax(-1))

                            # Calculate total error
                            error /= k_folds

                            # Check if these hyper parameters give us the smallest error
                            if min_error > error:
                                min_error = error
                                best_num_layers = num_layers
                                best_num_layer_units = num_layer_units
                                best_moment_scale = moment_scale
                                best_weight_scale = weight_scale

                            # Update weight scale
                            weight_scale *= weight_scale_grid['step']
                            # Check exit condition
                            if weight_scale >= weight_scale_grid['max_val']:
                                break

                        # Update moment scale
                        moment_scale *= moment_scale_grid['step']
                        # Check exit condition
                        if moment_scale >= moment_scale_grid['max_val']:
                            break

                    # Update number of layer units
                    num_layer_units *= num_layer_units_grid['step']
                    # Check exit condition
                    if num_layer_units >= num_layer_units_grid['max_val']:
                        break
                    
                # Update number of layers
                num_layers *= num_layers_grid['step']
                # Check exit condition
                if num_layers >= num_layers_grid['max_val']:
                    break

                # Set model hyper paramters
                best_layer_sizes = np.int32([num_features]+[best_num_layer_units for i in range(best_num_layers)]+[self.num_classes])
                self.model.setLayerSizes(best_layer_sizes)
                self.model.setTrainMethod(params['train_method'])
                self.model.setBackpropMomentumScale(best_moment_scale)
                self.model.setBackpropWeightScale(best_weight_scale)
                self.model.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 0.01))
                self.model.setActivationFunction(params['activation_func'], 2, 1)
                self.model.train(samples, cv2.ml.ROW_SAMPLE, np.float32(new_responses))

        # Optimize for RProp algorithm 
        elif params['train_method'] == cv2.ml.ANN_MLP_RPROP:
            num_samples, num_features = samples.shape
            new_responses = self.unroll_responses(responses).reshape(-1, self.num_classes)

            # Hyperparameters to optimize
            num_layers = 0
            num_layer_units = 0
            rp_dw0 = 0.0
            rp_dw_plus = 0.0
            rp_dw_minus = 0.0
            rp_dw_min = 0.0
            rp_dw_max = 0.0

            # Optimized hyperparameter values
            best_num_layers = 0
            best_num_layer_units = 0
            best_rp_dw0 = 0.0
            best_rp_dw_plus = 0.0
            best_rp_dw_minus = 0.0
            best_rp_dw_min = 0.0
            best_rp_dw_max = 0.0

            # The error to minimize
            min_error = float('inf')

            # Ensure sufficient cross-validation size
            if k_folds < 2:
                print 'K_folds value must be >= 2'
                exit(1)

            # Get default grids
            num_layers_grid = get_default_grid('NUM_LAYERS')
            num_layer_units_grid = get_default_grid('NUM_LAYER_UNITS')
            rp_dw0_grid = get_default_grid('RP_DW0')
            rp_dw_plus_grid = get_default_grid('RP_DW_PLUS')
            rp_dw_minus_grid = get_default_grid('RP_DW_MINUS')
            rp_dw_min_grid = get_default_grid('RP_DW_MIN')
            rp_dw_max_grid = get_default_grid('RP_DW_MAX')

            # Optimize hyper parameters
            num_layers = num_layers_grid['min_val']
            while True:
                num_layer_units = num_layer_units_grid['min_val']
                while True:
                    rp_dw0 = rp_dw0_grid['min_val']
                    while True:
                        rp_dw_plus = rp_dw_plus_grid['min_val']
                        while True:
                            rp_dw_minus = rp_dw_mins_grid['min_val']
                            while True:
                                rp_dw_min = rp_dw_min_grid['min_val']
                                while True:
                                    rp_dw_max = rp_dw_max_grid['min_val']
                                    while True:
                                        error = 0
                                        for traincv, testcv in sklearn.cross_validation.KFold(len(samples), n_folds=k_folds):
                                            # Set model hyper parameters
                                            layer_sizes = np.int32([num_features]+[num_layer_units for i in range(num_layers)]+[self.num_classes])
                                            self.model.setLayerSizes(layer_sizes)
                                            self.model.setTrainMethod(params['train_method'])
                                            self.model.setRpropDW(rp_dw0)
                                            self.model.setRpropDWPlus(rp_dw_plus)
                                            self.model.setRpropDWMinus(rp_dw_minus)
                                            self.model.setRpropDWMin(rp_dw_min)
                                            self.model.setRpropDWMax(rp_dw_max)
                                            self.model.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 0.01))
                                            self.model.setActivationFunction(params['activation_func'], 2, 1)

                                            # Train the model
                                            self.model.train(samples[traincv], cv2.ml.ROW_SAMPLE, np.float32(new_responses[traincv]))

                                            # Test the model
                                            ret, test_predict = self.model.predict(samples[testcv])

                                            # Sum over the error
                                            error += 1-1.0*sum([1 for label, predict in zip(new_responses[testcv].argmax(-1), test_predict.argmax(-1)) if label == predict]) / len(test_predict.argmax(-1))

                                            # Calculate total error
                                            error /= k_folds

                                            # Check if these hyper parameters give us the smallest error
                                            if min_error > error:
                                                min_error = error
                                                best_num_layers = num_layers
                                                best_num_layer_units = num_layer_units
                                                best_rp_dw0 = rp_dw0
                                                best_rp_dw_plus = rp_dw_plus
                                                best_rp_dw_minus = rp_dw_minus
                                                best_rp_dw_min = rp_dw_min
                                                best_rp_dw_max = rp_dw_max
  
                                        # Update
                                        rp_dw_max *= rp_dw_max_grid['step']
                                        # Check exit condition
                                        if rp_dw_max >= rp_dw_max_grid['max_val']:
                                            break

                                    # Update
                                    rp_dw_min *= rp_dw_min_grid['step']
                                    # Check exit condition
                                    if rp_dw_min >= rp_dw_min_grid['max_val']:
                                        break

                                # Update
                                rp_dw_minus *= rp_dw_minus_grid['step']
                                # Check exit condition
                                if rp_dw_minus >= rp_dw_minus_grid['max_val']:
                                    break

                            # Update
                            rp_dw_plus *= rp_dw_plus_grid['step']
                            # Check exit condition
                            if rp_dw_plus >= rp_dw_plus_grid['max_val']:
                                break

                        # Update
                        rp_dw0 *= rp_dw0_grid['step']
                        # Check exit condition
                        if rp_dw0 >= rp_dw0_grid['max_val']:
                            break

                    # Update number of layer units
                    num_layer_units *= num_layer_units_grid['step']
                    # Check exit condition
                    if num_layer_units >= num_layer_units_grid['max_val']:
                        break
                    
                # Update number of layers
                num_layers *= num_layers_grid['step']
                # Check exit condition
                if num_layers >= num_layers_grid['max_val']:
                    break

                # Set model hyper paramters
                best_layer_sizes = np.int32([num_features]+[best_num_layer_units for i in range(best_num_layers)]+[self.num_classes])
                self.model.setLayerSizes(best_layer_sizes)
                self.model.setTrainMethod(params['train_method'])
                self.model.setRpropDW(best_rp_dw0)
                self.model.setRpropDWPlus(best_rp_dw_plus)
                self.model.setRpropDWMinus(best_rp_dw_minus)
                self.model.setRpropDWMin(best_rp_dw_min)
                self.model.setRpropDWMax(best_rp_dw_max)
                self.model.train(samples, cv2.ml.ROW_SAMPLE, np.float32(new_responses))
                self.model.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 0.01))
                self.model.setActivationFunction(params['activation_func'], 2, 1)


    def predict(self, samples):
        """ 
            Perform the prediction of some samples using the trained model.

            Parameters
            ----------
            samples: numpy array
                The data samples to be predicted
        """
        ret, resp = self.model.predict(samples)
        return resp.argmax(-1)

