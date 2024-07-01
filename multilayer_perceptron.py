import numpy as np
from utils import identity, sigmoid, tanh, relu, softmax, cross_entropy, one_hot_encoding,cross_entropy_dev


class MultilayerPerceptron:
    """
    A class representing the machine learning implementation of a Multilayer Perceptron classifier from scratch.

    Attributes:
        n_hidden
            An integer representing the number of neurons in the one hidden layer of the neural network.

        hidden_activation
            A string representing the activation function of the hidden layer. The possible options are
            {'identity', 'sigmoid', 'tanh', 'relu'}.

        n_iterations
            An integer representing the number of gradient descent iterations performed by the fit(X, y) method.

        learning_rate
            A float representing the learning rate used when updating neural network weights during gradient descent.

        _output_activation
            An attribute representing the activation function of the output layer. This is set to the softmax function
            defined in utils.py.

        _loss_function
            An attribute representing the loss function used to compute the loss for each iteration. This is set to the
            cross_entropy function defined in utils.py.

        _loss_history
            A Python list of floats representing the history of the loss function for every 20 iterations that the
            algorithm runs for. The first index of the list is the loss function computed at iteration 0, the second
            index is the loss function computed at iteration 20, and so on and so forth. Once all the iterations are
            complete, the _loss_history list should have length n_iterations / 20.

        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model. This
            is set in the _initialize(X, y) method.

        _y
            A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the
            input data used when fitting the model.

        _h_weights
            A numpy array of shape (n_features, n_hidden) representing the weights applied between the input layer
            features and the hidden layer neurons.

        _h_bias
            A numpy array of shape (1, n_hidden) representing the weights applied between the input layer bias term
            and the hidden layer neurons.

        _o_weights
            A numpy array of shape (n_hidden, n_outputs) representing the weights applied between the hidden layer
            neurons and the output layer neurons.

        _o_bias
            A numpy array of shape (1, n_outputs) representing the weights applied between the hidden layer bias term
            neuron and the output layer neurons.

    Methods:
        _initialize(X, y)
            Function called at the beginning of fit(X, y) that performs one-hot encoding for the target class values and
            initializes the neural network weights (_h_weights, _h_bias, _o_weights, and _o_bias).

        fit(X, y)
            Fits the model to the provided data matrix X and targets y.

        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """

    def __init__(self, n_hidden = 16, hidden_activation = 'sigmoid', n_iterations = 1000, learning_rate = 0.01):
        # Create a dictionary linking the hidden_activation strings to the functions defined in utils.py
        activation_functions = {'identity': identity, 'sigmoid': sigmoid, 'tanh': tanh, 'relu': relu}

        # Check if the provided arguments are valid
        if not isinstance(n_hidden, int) \
                or hidden_activation not in activation_functions \
                or not isinstance(n_iterations, int) \
                or not isinstance(learning_rate, float):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the MultilayerPerceptron model object
        self.n_hidden = n_hidden
        self.hidden_activation = activation_functions[hidden_activation]
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self._output_activation = softmax
        self._loss_function = cross_entropy
        self._loss_history = []
        self._X = None
        self._y = None
        self._h_weights = None
        self._h_bias = None
        self._o_weights = None
        self._o_bias = None

    def _initialize(self, X, y):
        """
        Function called at the beginning of fit(X, y) that performs one hot encoding for the target class values and
        initializes the neural network weights (_h_weights, _h_bias, _o_weights, and _o_bias).

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """

        self._X = X
        self.samp, self.feature = np.shape(X)
        #print("here",self.total_outputs)
        self._y = one_hot_encoding(y)
        # print("sdkufsdf",np.shape(self._y))
        self.total_outputs= np.shape(self._y)[1]
        # self._y=y

        #initialising weights
        lim =1/self.feature**0.5 
        
        #Input Layer
        self._h_weights= np.random.uniform(-lim,lim, (self.feature, self.n_hidden))
        #print(np.shape(self._h_weights))
        self._h_bias= np.zeros((1, self.n_hidden))

        #output Layer
        self._o_weights= np.random.uniform(-lim,lim, (self.n_hidden,self.total_outputs))
        self._o_bias= np.zeros((1,self.total_outputs))

        np.random.seed(42)

    def fit(self, X, y):
        """
        Fits the model to the provided data matrix X and targets y and stores the cross-entropy loss every 20
        iterations.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """

        self._initialize(X, y)
        iteration=0
        while(iteration<self.n_iterations):
            #for i,j in enumerate(self._X):
                #input to hidden layer
            #print("j",np.shape(j))
            
            #print(np.shape(self._h_weights))
            #print(np.shape(X))

            hid_layer_in= np.dot(X,self._h_weights) + self._h_bias
            #print(np.shape(hid_layer_in))
            #print(self.hidden_activation)
            hid_layer_out= self.hidden_activation(hid_layer_in,derivative=False)
            #print("hid_out", np.shape(hid_layer_out))
            #print("o_out", np.shape(self._o_weights))
            out_layer_in = np.dot(hid_layer_out,self._o_weights)+self._o_bias
            #print(np.shape(out_layer_in))
            out_layer_out= self._output_activation(out_layer_in,derivative=False)
            #print(out_layer_out)

            #back_propagation
            error_21= cross_entropy_dev(np.array(self._y),out_layer_out)*self._output_activation(out_layer_in,derivative=True)
            #print("shapee21",np.shape(error_21))
            error_2= np.dot(hid_layer_out.T,error_21)
            error_till_2= np.sum(error_21,axis=0,keepdims=True)
            #print("o_weit", np.shape(self._o_weights))

            error_till_1_temp= np.dot(error_21,self._o_weights.T) * self.hidden_activation(hid_layer_in)
            error_1= np.dot(X.T,error_till_1_temp)
            error_till_1= np.sum(error_1,axis=0,keepdims=True)


            self._o_weights= self._o_weights- self.learning_rate*error_2
            self._o_bias= self._o_bias- self.learning_rate* error_till_2

            self._h_weights= self._h_weights- self.learning_rate*error_1
            self._h_bias= self._h_bias- self.learning_rate*error_till_1

        iteration+=1





    def predict(self, X):
        """
        Predicts class target values for the given test data matrix X using the fitted classifier model.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the test data.

        Returns:
            A numpy array of shape (n_samples,) representing the predicted target class values for the given test data.
        """

        hid_in = X.dot(self._h_weights) + self._h_bias
        hid_out = self.hidden_activation(hid_in)
        out_in = hid_out.dot(self._o_weights) + self._o_bias
        y_pred = self._output_activation(out_in)
        
        return y_pred

        




