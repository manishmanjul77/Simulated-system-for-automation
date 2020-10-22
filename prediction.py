import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    s = 1 / (1 + np.exp(-z))
    ### END CODE HERE ###
    
    return s
def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    ### START CODE HERE ### (≈ 1 line of code)
    A = sigmoid(w.T @ X + b) 
    ### END CODE HERE ###
    
    for i in range(A.shape[1]):
        
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        ### START CODE HERE ### (≈ 4 lines of code)
        Y_prediction[:, i] = (A[:, i] > 0.5) * 1
        ### END CODE HERE ###
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction
def cool(my_image):
    num_px=64

    #my_image ="download.jpg"   # change this to the name of your image file
    ## END CODE HERE ##

    # We preprocess the image to fit your algorithm.
    fname = "media/" + my_image
    image=Image.open(fname).resize(size=(num_px,num_px))
    my_image = np.array(image, dtype=float) / 255  # convert to numpy array and scale values
    my_image = my_image.reshape((1, num_px*num_px*3)).T
    wih=np.load('weight.npy')
    b=np.load('weightb.npy')
    my_predicted_image = predict(wih, b, my_image)
    return np.squeeze(my_predicted_image)


    #print("y "+str(np.squeeze(my_predicted_image)))

