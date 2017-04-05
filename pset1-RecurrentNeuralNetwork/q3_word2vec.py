import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function """
    # Implement a function that normalizes each row of a matrix to have unit length
    
    ### YOUR CODE HERE
    x = x / np.sqrt(np.sum(x * x, axis=1, keepdims=True) + 1e-30)
    ### END YOUR CODE
    
    return x

def test_normalize_rows():
    print ("Testing normalizeRows...")
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]])) 
    # the result should be [[0.6, 0.8], [0.4472, 0.8944]]
    print (x)
    assert (x.all() == np.array([[0.6, 0.8], [0.4472, 0.8944]]).all())
    print ("")

def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models """
    
    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, assuming the softmax prediction function and cross      
    # entropy loss.                                                   
    
    # Inputs:                                                         
    # - predicted: numpy ndarray, predicted word vector (\hat{v} in 
    #   the written component or \hat{r} in an earlier version)
    # - target: integer, the index of the target word               
    # - outputVectors: "output" vectors (as rows) for all tokens     
    # - dataset: needed for negative sampling, unused here.         
    
    # Outputs:                                                        
    # - cost: cross entropy cost for the softmax word prediction    
    # - gradPred: the gradient with respect to the predicted word   
    #        vector                                                
    # - grad: the gradient with respect to all the other word        
    #        vectors                                               
    
    
    ### YOUR CODE HERE
    #print ("==softmax======================")
    #print ("target", target)
    #print ("predicted.shape",predicted.shape)
    #print ("predicted",predicted)
    #print ("outputVectors.shape",outputVectors.shape)
    #print ("outputVectors",outputVectors)
    
    #z = outputVectors.dot(predicted)
    #z = predicted.dot(outputVectors.T)
    #y_hat = softmax(z)
    #cost = - np.log(y_hat[target])
    #print ("y_hat",y_hat)
    #print ("softmax cost:",cost)

    # delta = y_hat - y
    #delta = y_hat
    #delta[target] -= 1
    
    #print ("delta[target] -= 1",delta)

    # get gradient shapes from the jacobian
    # N is the size of the vocabulary (your output dim)
    # D is the size of the word vector
    #N = delta.shape[0]
    #D = predicted.shape[0]

    # outputVectors = U
    # dJ/dv_c = delta*U^T - tranpose of jacobian
    #gradPred = delta.dot(outputVectors)
    #print ("softmax gradPred:",gradPred)
    
    # gradient wrt all other word vectors
    # dJ/du_w = delta * predicted^T
    #grad = np.outer(delta, predicted)
    #print ("softmax grad:",grad)
    
    probabilities = softmax(predicted.dot(outputVectors.T))
    cost = -np.log(probabilities[target])
    delta = probabilities
    delta[target] -= 1
    N = delta.shape[0]
    D = predicted.shape[0]
    grad = delta.reshape((N,1)) * predicted.reshape((1,D))
    gradPred = (delta.reshape((1,N)).dot(outputVectors)).flatten()
    
    ### END YOUR CODE
    
    return cost, gradPred, grad

def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, 
    K=10):
    """ Negative sampling cost function for word2vec models """

    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, using the negative sampling technique. K is the sample  
    # size. You might want to use dataset.sampleTokenIdx() to sample  
    # a random word index. 
    # 
    # Note: See test_word2vec below for dataset's initialization.
    #                                       
    # Input/Output Specifications: same as softmaxCostAndGradient     
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!
    
    ### YOUR CODE HERE
    #print ("==neg-sampling======================")
    
    # initialize gradients since updates will be sparse
    #gradPred = np.zeros(predicted.shape)
    #grad = np.zeros(outputVectors.shape)

    # log(sigmoid(u_o^T v_c))
    #z_pos = outputVectors[target].dot(predicted)
    #p_pos = sigmoid(z_pos)
    #log_p_pos = np.log(p_pos)

    # Sum over K of log(sigmoid(-u_k^T v_c))
    #sum_neg = 0
    #ns_idxs = np.zeros(K)
    #neg_sample_deltas = np.zeros(K)
    #for i in range(K):
    #    ns_idx = dataset.sampleTokenIdx()
    #    while ns_idx == target:
    #        ns_idx = dataset.sampleTokenIdx()
    #    z_neg = - outputVectors[ns_idx].dot(predicted)
    #    p_neg = sigmoid(z_neg)
    #    log_p_neg = np.log(p_neg)

    #    sum_neg += log_p_neg
    #    # bookeeping for gradients
    #    ns_idxs[i] = ns_idx
    #    neg_sample_deltas[i] = p_neg - 1

    #cost = - log_p_pos - sum_neg

    # dJ/dv_c
    # (sigmoid(u_o^T v_c) - 1) u_o
    #delta_pos = p_pos - 1
    #gradPred += delta_pos * outputVectors[target]
    #for i in range(K):
    #    gradPred -= neg_sample_deltas[i] * outputVectors[ns_idxs[i]]

    # dJ/du_o
    #grad[target] = delta_pos * predicted

    # dJ/du_k
    #for i in range(K):
    #   grad[ns_idxs[i]] -= neg_sample_deltas[i] * predicted

    grad = np.zeros(outputVectors.shape)
    gradPred = np.zeros(predicted.shape)
    
    indices = [target]
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices += [newidx]
        
    labels = np.array([1] + [-1 for k in range(K)])
    vecs = outputVectors[indices,:]
    
    t = sigmoid(vecs.dot(predicted) * labels)
    cost = -np.sum(np.log(t))
    
    delta = labels * (t - 1)
    gradPred = delta.reshape((1,K+1)).dot(vecs).flatten()
    gradtemp = delta.reshape((K+1,1)).dot(predicted.reshape(
        (1,predicted.shape[0])))
    for k in range(K+1):
        grad[indices[k]] += gradtemp[k,:]

    
    ### END YOUR CODE
    
    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors, 
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ Skip-gram model in word2vec """

    # Implement the skip-gram model in this function.

    # Inputs:                                                         
    # - currrentWord: a string of the current center word           
    # - C: integer, context size                                    
    # - contextWords: list of no more than 2*C strings, the context words                                               
    # - tokens: a dictionary that maps words to their indices in    
    #      the word vector list                                
    # - inputVectors: "input" word vectors (as rows) for all tokens           
    # - outputVectors: "output" word vectors (as rows) for all tokens         
    # - word2vecCostAndGradient: the cost and gradient function for 
    #      a prediction vector given the target word vectors,  
    #      could be one of the two cost functions you          
    #      implemented above

    # Outputs:                                                        
    # - cost: the cost function value for the skip-gram model       
    # - grad: the gradient with respect to the word vectors         
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!

    ### YOUR CODE HERE
    print ("==skipgram==========================================================")

    #cost = 0
    #gradIn = np.zeros(inputVectors.shape)
    #gradOut = np.zeros(outputVectors.shape)

    # we are gathering input_word, its index in vocabulary and its feature vector
    #current_word_index = tokens[currentWord]
    #current_word_vector = inputVectors[current_word_index]

    
    # iterating thru each of the context words that are predicted
    #for contextWord in contextWords:
        # get target word index from vocabulary
    #    context_word_index = tokens[contextWord]
    #    c_cost, c_grad_in, c_grad_out = word2vecCostAndGradient(current_word_vector, context_word_index, outputVectors, dataset)
    #    cost += c_cost
    #    gradIn[context_word_index] += c_grad_in
    #   gradOut += c_grad_out

    print("\n inputVectors:\n",inputVectors)
    print("\n outputVectors:\n",outputVectors)
    currentI = tokens[currentWord]
    predicted = inputVectors[currentI, :]    
    print("skipgram currentWord", currentWord)
    print("skipgram predicted", predicted)
    print("skipgram contextWords", contextWords)
    
    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)
    for contextWord in contextWords:
        #idx = tokens[contextWord]
        print("skipgram loop contextWord", contextWord," tokens[contextWord]", tokens[contextWord] )
        cc, gp, gg = word2vecCostAndGradient(predicted, tokens[contextWord], outputVectors, dataset)
        cost += cc
        gradOut += gg
        gradIn[currentI, :] += gp
   
    ### END YOUR CODE
    
    return cost, gradIn, gradOut

def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors, 
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ CBOW model in word2vec """

    # Implement the continuous bag-of-words model in this function.            
    # Input/Output specifications: same as the skip-gram model        
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!

    #################################################################
    # IMPLEMENTING CBOW IS EXTRA CREDIT, DERIVATIONS IN THE WRIITEN #
    # ASSIGNMENT ARE NOT!                                           #  
    #################################################################
    
    cost = 0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE

    print ("CBOW but not mine ;) ")
    predicted = np.zeros(inputVectors.shape[1])

    print("\n inputVectors:\n",inputVectors)
    print("\n outputVectors:\n",outputVectors)
    
    # Sum up input context words \hat v = \sum_{-m <= j <= m, j != 0} v_{c+j}
    print("cbow contextWords", contextWords)
    for contextWord in contextWords:
        idx = tokens[contextWord]
        predicted += inputVectors[idx]
        print("cbow loop inputVectors[idx]", inputVectors[idx]," tokens[contextWord]", tokens[contextWord])
        print("cbow loop contextWord", contextWord," predicted", predicted)

    print("cbow currentWord", currentWord," tokens[currentWord]", tokens[currentWord])
    cost, gradPredict, gradOut = word2vecCostAndGradient(predicted, tokens[currentWord], outputVectors, dataset)

    # Collect gradient for all the input context words
    for contextWord in contextWords:
        idx = tokens[contextWord]
        gradIn[idx] += gradPredict
        
    ### END YOUR CODE
    
    return cost, gradIn, gradOut

#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient = softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in range(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)
        
        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1
        
        c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom
        
    return cost, grad

def test_word2vec():
    # Interface to the dataset for negative sampling
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] \
           for i in range(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print ("==== Gradient check for skip-gram ====")
    #gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)
    print ("\n==== Gradient check for CBOW      ====")
    #gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)

    print ("\n=============================================")
    print ("\n=============================================")
    print ("\n=== Results ===")
    print ("\n=============================================")
    print ("\n=============================================")
    print (skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    print (skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient))
    print (cbow("a", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    print (cbow("a", 2, ["a", "b", "a", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient))

if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()