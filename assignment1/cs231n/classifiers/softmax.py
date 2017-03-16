import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

    
  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  scores  = X.dot(W)
  exp_sco = np.exp(scores)
  dL_dsco = np.zeros_like(exp_sco)
  
  for i in xrange(num_train):
    
    exp_sum = 0
    
    for j in xrange(num_classes):
        
        exp_sum += exp_sco[i,j]
            
    exp_fra = exp_sco[i,y[i]] / exp_sum
    loss += -np.log(exp_fra)
    
    dL_dfra = -1/ exp_fra#(-np.log(exp_fra))
    
    for j in xrange(num_classes):
        dL_dsco[i,j] = (-exp_fra *  (exp_sco[i,j] / exp_sum)) * dL_dfra
        
        if j == y[i]:
            dL_dsco[i,j] = (exp_fra - np.power(exp_fra, 2)) * dL_dfra
        
  dW = X.transpose().dot(dL_dsco)


  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW   /= num_train 
  
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  num_classes = W.shape[1]
  num_train = X.shape[0]  
  
  # calculate the score matrix
  scores  = X.dot(W)
  
  # calculate the exponential fractions
  exp_sco = np.exp(scores)
  exp_sum = np.sum(exp_sco,1)
  exp_fra = np.divide(exp_sco, exp_sum[:,None])
  los_fra = exp_fra[np.arange(num_train),y]
 

    
  # calculate the loss:
  loss = np.sum(-np.log(los_fra))

    
    
  # calculate the gradient:
  # initialisation
  dL_dfra = np.zeros_like(los_fra)
  dL_dsco = np.zeros_like(exp_sco)
    
  # dL_dfra: calculate the derivative of L to the fraction of correct y indices
  dL_dfra = np.divide(-1,los_fra)
    
  # dL_dsco: calculate the derivative of the fraction to calculated 'scores' matrix, times dL_dfra
  mult    = np.multiply(-los_fra, dL_dfra)
  dL_dsco = np.multiply(exp_fra, mult[:,None])
     
  # dL_dsco: correction for the scores with correct indices 
  dL_dsco[np.arange(num_train),y] = np.multiply(np.subtract(los_fra, np.power(los_fra, 2)), dL_dfra)
  
  # dW: calculate the derivative of the calculated 'scores' matrix to the weight matrix W, times dl_dsco
  dW = X.transpose().dot(dL_dsco)



  # correct loss and gradient for average and regularization:
  # average
  loss /= num_train
  dW   /= num_train 
  
  # regularization
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W  
        
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

