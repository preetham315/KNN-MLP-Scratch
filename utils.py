import numpy as np


def euclidean_distance(x1, x2):
    """
    Computes and returns the Euclidean distance between two vectors.

    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    """
    dist_e =0
    for p in range(len(x1)):

        #dist_e= np.sqrt(np.sum(np.square(x1[p]-x2[p])))
        dist_e += (x1[p]-x2[p])**2
        #print(dist_e)
    return (dist_e)**0.5


def manhattan_distance(x1, x2):
    """
    Computes and returns the Manhattan distance between two vectors.

    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    """
    dist_m=0
    for p in range(len(x1)):
        dist_m+= abs(x1[p]-x2[p])
    return dist_m


def identity(x, derivative = False):
    """
    Computes and returns the identity activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    
    if derivative:
        return 1
    else:
        return x



def sigmoid(x, derivative = False):
    """
    Computes and returns the sigmoid (logistic) activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    if derivative:
        return np.exp(-x)/(1+np.exp(-x))**2
    else:
        print("here S")
        return 1/1+np.exp(-x)


def tanh(x, derivative = False):
    """
    Computes and returns the hyperbolic tangent activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    if derivative:
        return 4/ (np.exp(x)+np.exp(-x))**2
    else:
        print("here T")
        return (np.exp(x) -np.exp(-x))/np.exp(x)+np.exp(-x)


def relu(x, derivative = False):
    """
    Computes and returns the rectified linear unit activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """

    if derivative:
        if x >0:
            return 1 
    else:
        print("here R")
        return max(0,x)


def softmax(x, derivative = False):
    x = np.clip(x, -1e100, 1e100)
    # print(np.shape(x))
    if not derivative:
        c = np.max(x, axis = 1, keepdims = True)
        return np.exp(x - c - np.log(np.sum(np.exp(x - c), axis = 1, keepdims = True)))
    else:
        return softmax(x) * (1 - softmax(x))


def cross_entropy(y, p):
    """
    Computes and returns the cross-entropy loss, defined as the negative log-likelihood of a logistic model that returns
    p probabilities for its true class labels y.

    Args:
        y:
            A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the
            input data used when fitting the model.
        p:
            A numpy array of shape (n_samples, n_outputs) representing the predicted probabilities from the softmax
            output activation function.
    """
    entropy= (- np.sum(y* np.log(p)))/float(np.shape(p)[0])
    return entropy

def cross_entropy_dev(y,p):
    #print(type(y))
    #print(type(p))
    p = np.clip(p, 1e-15, 1 - 1e-15)
    # print("p is",1-p)
    return - y/ p + (1 - y)/(1 - p)


def one_hot_encoding(y):
    """
    Converts a vector y of categorical target class values into a one-hot numeric array using one-hot encoding: one-hot
    encoding creates new binary-valued columns, each of which indicate the presence of each possible value from the
    original data.

    Args:
        y: A numpy array of shape (n_samples,) representing the target class values for each sample in the input data.

    Returns:
        A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the input
        data. n_outputs is equal to the number of unique categorical class values in the numpy array y.
    """
    y_unique= np.unique(y)
    temp_dict={}
    for i in range(len(y_unique)):
        temp_dict[y_unique[i]]= i
    encoded=[]
    A_init= list(np.zeros(len(y_unique), dtype= int))
    for i in y:
        A_init[temp_dict[i]]=1
        encoded.append(A_init)
    return encoded


