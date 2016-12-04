import numpy as np
import random

def softmax(x):
    """
    Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code.
    You might find numpy functions np.exp, np.sum, np.reshape,
    np.max, and numpy broadcasting useful for this task. (numpy
    broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

    You should also make sure that your code works for one
    dimensional inputs (treat the vector as a row), you might find
    it helpful for your later problems.

    You must implement the optimization in problem 1(a) of the 
    written assignment!
    
    a :
        [[1001 1002]
         [   3    4]]
    a.shape : (2, 2)
    len(a.shape) : 2
    np.max(a, axis = 0, keepdims=True) :
        [[1002]
         [   4]]
    b :
        [1001 1002]
    b.shape : (2,)
    len(b.shape) : 1
    
    c = b.reshape(1, b.shape[0]) :
        [[1001 1002]]
    c.shape : (1, 2)
    
    """

    ### YOUR CODE HERE
    
    #transform Nx1 in 1xN (1 row x N cols) else keep NxM
    ##if len(x.shape) == 1:
    ##    x = x.reshape(1, x.shape[0])
    ##
    ##  # Compute the actual softmax function
    ## # np.max(x, axis = 1, keepdims=True) give max value for each row
    ## xexp = np.exp(x - np.max(x, axis = 1, keepdims=True))
    ## x = (xexp / np.sum(xexp, 1))

    if len(x.shape) > 1:
        tmp = np.max(x, axis = 1)
        x -= tmp.reshape((x.shape[0], 1))
        x = np.exp(x)
        tmp = np.sum(x, axis = 1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp
    
    ### END YOUR CODE
    
    return x

def test_softmax_basic():
    """
    Some simple tests to get you started. 
    Warning: these are not exhaustive.
    """
    print ("Running basic tests...")
    test1 = softmax(np.array([1,2]))
    print (test1)
    assert np.amax(np.fabs(test1 - np.array(
        [0.26894142,  0.73105858]))) <= 1e-6

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print (test2)
    assert np.amax(np.fabs(test2 - np.array(
        [[0.26894142, 0.73105858], [0.26894142, 0.73105858]]))) <= 1e-6

    test3 = softmax(np.array([[-1001,-1002]]))
    print (test3)
    assert np.amax(np.fabs(test3 - np.array(
        [0.73105858, 0.26894142]))) <= 1e-6

    print ("You should verify these results!\n")

def test_softmax():
    """ 
    Use this space to test your softmax implementation by running:
        python q1_softmax.py 
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print ("Running your tests...")
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE  

if __name__ == "__main__":
    test_softmax_basic()
    test_softmax()