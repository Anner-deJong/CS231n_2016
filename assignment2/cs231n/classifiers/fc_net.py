import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    
    # weight matrices
    self.params['W1'] = np.random.randn(input_dim,  hidden_dim ) * weight_scale
    self.params['W2'] = np.random.randn(hidden_dim, num_classes) * weight_scale
    
    # bias arrays
    self.params['b1'] = np.zeros(hidden_dim )
    self.params['b2'] = np.zeros(num_classes)
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    
    # hidden layer
    a, hid_cache    = affine_relu_forward(X, self.params['W1'], self.params['b1'])

    # output layer
    scores, out_cache = affine_forward(a, self.params['W2'], self.params['b2'])
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    
    # pre calculate regularization
    reg_w1   = self.reg * self.params['W1']
    reg_w1_q = np.sum(np.multiply(reg_w1, self.params['W1']))
    reg_w2   = self.reg * self.params['W2']
    reg_w2_q = np.sum(np.multiply(reg_w2, self.params['W2']))
    reg_loss = 0.5 * (reg_w1_q + reg_w2_q)
    
    # backpropagate
    
    # softmax loss and d_output
    sof_loss, d_out = softmax_loss(scores, y)
    loss            = sof_loss + reg_loss 
    
    # output later, W2, b2 
    d_a, dw, db     = affine_backward(d_out, out_cache)
    grads['W2']     = dw + reg_w2
    grads['b2']     = db
    
    # hidden layer, W1, b1
    _, dw, db       = affine_relu_backward(d_a, hid_cache)
    grads['W1']     = dw + reg_w1
    grads['b1']     = db
  
 #  # softmax loss and d_output
 #  sof_loss, d_out = softmax_loss(scores, y)
 #  loss            = sof_loss + reg_loss 
 #  
 #  # d_activated hidden layer, W2, b2
 #  grads['W2']     = dw + reg_w2
 #  grads['b2']     = db
 #  
 #  # d_unactivated hidden layer
 #  d_c             = relu_backward(d_a, relu_cache)
 #
 #  # d_input, W1, b1
 #  grads['W1']     = dw + reg_w1
 #  grads['b1']     = db #  
    
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers determinstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    
    ### no hidden layer ###
    if self.num_layers == 1:
        first_hid_out_size = num_classes
    
    
    ### output layer (if at least one hidden layer) ###
    else:
        first_hid_out_size = hidden_dims[0]
        
        # weight matrix
        out_wei = ('W%d' % self.num_layers)
        self.params[out_wei] = np.random.randn(hidden_dims[self.num_layers-2], num_classes) * weight_scale    
            
        # bias array
        out_bia = ('b%d' % self.num_layers)
        self.params[out_bia] = np.zeros(num_classes)

    
    ### first hidden layer ###   
    
    # weight matrix
    self.params['W1'] = np.random.randn(input_dim,  first_hid_out_size) * weight_scale    
        
    # bias array
    self.params['b1'] = np.zeros(first_hid_out_size)
    
    #if self.use_batchnorm:
    # scale array
    self.params['gamma1'] = np.ones(first_hid_out_size)
    
    # shift array
    self.params['beta1'] = np.zeros(first_hid_out_size)
    
    
    ### hidden layer (except first) ###
    if self.num_layers > 2:
        for lay_no in xrange(2, self.num_layers):
            
            # weight matrices
            cur_wei = ('W%d' % lay_no)
            self.params[cur_wei] = np.random.randn(hidden_dims[lay_no-2],  hidden_dims[lay_no-1]) * weight_scale
            
            # bias arrays
            cur_bia = ('b%d' % lay_no)
            self.params[cur_bia] = np.zeros(hidden_dims[lay_no-1])
            
            # if self.use_batchnorm:
            # scale array
            cur_gam = ('gamma%d' % lay_no)
            self.params[cur_gam] = np.ones(hidden_dims[lay_no-1])
            
            # shift array
            cur_bet = ('beta%d' % lay_no)
            self.params[cur_bet] = np.zeros(hidden_dims[lay_no-1])
    
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    #if self.use_batchnorm:
    self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    #if self.use_batchnorm:
    for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    
    # create dictionary to store caches
    caches   = {} 
    
    # initialise first input
    cur_inp     = X
    
      
    # hidden layers without batch normalisation    
    #if not self.use_batchnorm:
    #    for lay_no in xrange(1, self.num_layers):
    #        
    #        cur_wei         = self.params[('W%d' % lay_no)]
    #        cur_bia         = self.params[('b%d' % lay_no)]
    #        cur_cac         = ('C%d' % lay_no)
    #        
    #        a, hid_cache    = affine_relu_forward(cur_inp, cur_wei, cur_bia)
    #        
    #        if self.use_dropout:
    #            cur_inp, drop_cache = dropout_forward(a, self.dropout_param)
    #            caches[cur_cac] = [hid_cache, drop_cache]
    #            
    #        else:
    #            cur_inp         = a
    #            caches[cur_cac] = hid_cache
    #
    #
    ## hidden layers with batch normalisation
    #else:   
    for lay_no in xrange(1, self.num_layers):
    
        cur_wei         = ('W%d' % lay_no)
        cur_bia         = self.params[('b%d' % lay_no)]
        cur_gam         = self.params[('gamma%d' % lay_no)]
        cur_bet         = self.params[('beta%d'  % lay_no)]
        cur_cac         = ('C%d' % lay_no)
        
        a, hid_cache    = affine_batchnorm_relu_forward(cur_inp, self.params[cur_wei], cur_bia, cur_gam, cur_bet, self.bn_params[lay_no -1], self.use_batchnorm)
        if self.use_dropout:
            cur_inp, drop_cache = dropout_forward(a, self.dropout_param)
            caches[cur_cac] = [hid_cache, drop_cache]
            
        else:
            cur_inp         = a
            caches[cur_cac] = hid_cache
            
    
    # output layer
    out_wei             = ('W%d' % self.num_layers)
    out_bia             = ('b%d' % self.num_layers)
    scores, out_cache   = affine_forward(cur_inp, self.params[out_wei], self.params[out_bia])    
    
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    
    # initialise zero reg
    reg_loss           = 0
    
    ### softmax loss and d_ final hidden layer ###  
    sof_loss, d_out    = softmax_loss(scores, y)
    d_a, dw, db        = affine_backward(d_out, out_cache)
    
    # regularization
    out_wei            = ('W%d' % self.num_layers)
    out_reg            = self.reg * self.params[out_wei]
    reg_loss          += 0.5 * np.sum(np.multiply(out_reg, self.params[out_wei]))

    grads[out_wei]     = dw + out_reg
    grads[out_bia]     = db
    
    
    # d_hidden layers without batch normalisation    
    #if not self.use_batchnorm:
    #    for lay_no in xrange(self.num_layers-1, 0, -1):
    #        
    #        cur_wei         = ('W%d' % lay_no)
    #        cur_bia         = ('b%d' % lay_no)
    #        cur_cac         = ('C%d' % lay_no)
    #        
    #        if self.use_dropout:
    #            hid_cache, drop_cache = caches[cur_cac]
    #            d_d                   = dropout_backward(d_a, drop_cache)
    #            
    #            
    #        else:
    #            hid_cache       = caches[cur_cac]
    #            d_d             = d_a
    #            
    #        dx, dw, db            = affine_relu_backward(d_d, hid_cache)
    #        d_a                   = dx
    #        
    #        cur_reg         = self.reg * self.params[cur_wei]
    #        reg_loss       += 0.5 * np.sum(np.multiply(cur_reg, self.params[cur_wei]))
    #        
    #        grads[cur_wei]  = dw + cur_reg
    #        grads[cur_bia]  = db
            
            
    # d_hidden layers with batch normalisation
    #else:   
    for lay_no in xrange(self.num_layers-1, 0, -1):
        
        cur_wei         = ('W%d' % lay_no)
        cur_bia         = ('b%d' % lay_no)
        cur_gam         = ('gamma%d' % lay_no)
        cur_bet         = ('beta%d'  % lay_no)
        cur_cac         = ('C%d' % lay_no)
        
        if self.use_dropout:
            hid_cache, drop_cache = caches[cur_cac]
            d_d                   = dropout_backward(d_a, drop_cache)       
            
        else:
            hid_cache       = caches[cur_cac]
            d_d             = d_a
        
        dx, dw, db, dgamma, dbeta = affine_batchnorm_relu_backward(d_d, hid_cache, self.use_batchnorm)
        d_a             = dx
        
        cur_reg         = self.reg * self.params[cur_wei]
        reg_loss       += 0.5 * np.sum(np.multiply(cur_reg, self.params[cur_wei]))
        
        grads[cur_wei]  = dw + cur_reg
        grads[cur_bia]  = db
        grads[cur_gam]  = dgamma
        grads[cur_bet]  = dbeta
        
    
    # include regularization
    loss            = sof_loss + reg_loss 
    
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
