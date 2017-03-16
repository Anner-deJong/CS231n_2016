import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    
    # key values
    C, H, W    = input_dim
    F, wH, wW  = num_filters, filter_size, filter_size
    
    # MAGIC NUMBER!? UPDATE
    hid_in_size = F * H/2 * W/2
    
    # weight matrices
    self.params['W1'] = np.random.randn(num_filters, C,  wH, wW ) * weight_scale
    self.params['W2'] = np.random.randn(hid_in_size, hidden_dim ) * weight_scale
    self.params['W3'] = np.random.randn(hidden_dim , num_classes) * weight_scale
    
    # bias arrays
    self.params['b1'] = np.zeros(num_filters)
    self.params['b2'] = np.zeros(hidden_dim )
    self.params['b3'] = np.zeros(num_classes)
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    
    # conv relu pool layer
    crp_out, crp_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    
    # hidden relu layer
    ar_out , ar_cache  = affine_relu_forward(crp_out, W2, b2)
    
    # output layer
    scores , a_cache   = affine_forward(ar_out, W3, b3)
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    
    # pre calculate regularization
    reg_W1   = self.reg * W1
    reg_W1_q = np.sum(np.multiply(reg_W1, W1))
    reg_W2   = self.reg * W2
    reg_W2_q = np.sum(np.multiply(reg_W2, W2))
    reg_W3   = self.reg * W3
    reg_W3_q = np.sum(np.multiply(reg_W3, W3))
    reg_loss = 0.5 * (reg_W1_q + reg_W2_q + reg_W3_q)
    
    # backpropagate
    
    # softmax loss and d_output
    sof_loss, d_out = softmax_loss(scores, y)
    loss            = sof_loss + reg_loss 
    
    # output later, W3, b3
    d_ar_out, d_W3, d_b3 = affine_backward(d_out, a_cache)
    grads['W3']          = d_W3 + reg_W3
    grads['b3']          = d_b3
    
    # hidden layer, W2, b2
    d_cpr_out, d_W2, d_b2 = affine_relu_backward(d_ar_out, ar_cache)
    grads['W2']           = d_W2 + reg_W2
    grads['b2']           = d_b2
    
    # conv relu pool layer, W1, b1
    _, d_W1, d_b1 = conv_relu_pool_backward(d_cpr_out, crp_cache)
    grads['W1']   = d_W1 + reg_W1
    grads['b1']   = d_b1
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass





class Conv_1(object):
  """
  An CN + AN convolutional network with the following architecture:
  
  [conv-relu-conv-relu-pool]xCN - [affine-relu]xAN - [affine] - [softmax]
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_conv=1, num_filters=46, filter_size=5,
               num_hidden=2, hidden_dim=100, num_classes=10,
               dropout=0, use_batchnorm=False, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    
    - num_conv:    Number of [conv-relu-conv-relu-pool] layer combinations
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    
    - num_hidden:  Number of affine layers to include
    - hidden_dim:  Number of units to use in the fully-connected hidden layers
    - num_classes: Number of scores to produce from the final affine layer.
    
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.C1_bn_pars = []
    self.C2_bn_pars = []
    self.A_bn_pars = []
    self.reg = reg
    self.dtype = dtype
    self.num_conv = num_conv
    self.num_hid  = num_hidden
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    
    # key values
    C, H, W    = input_dim
    CN, AN     = num_conv, num_hidden
    F, wH, wW  = num_filters, filter_size, filter_size
    
    assert CN >= 1
    assert AN >= 1
    
    #################### [conv-relu-conv-relu-pool]xCN  weights and biases ####################
    
    # first conv layer combination
    self.params['CW11'] = np.random.randn(F, C, wH, wW ) * weight_scale
    self.params['CW12'] = np.random.randn(F, F, wH, wW ) * weight_scale
    self.params['Cb11'] = np.zeros(num_filters)
    self.params['Cb12'] = np.zeros(num_filters)
    self.params['Cgam11'] = np.ones(F)
    self.params['Cgam12'] = np.ones(F)
    self.params['Cbet11'] = np.zeros(F)
    self.params['Cbet12'] = np.zeros(F)
    
    # other conv layer combinations, if any
    for i in xrange(2, CN+1):
        
        cur_W_1 = ('CW%d1' % i)
        cur_W_2 = ('CW%d2' % i)
        cur_b_1 = ('Cb%d1' % i)
        cur_b_2 = ('Cb%d2' % i)
        
        self.params[cur_W_1] = np.random.randn(F, F, wH, wW ) * weight_scale
        self.params[cur_b_1] = np.zeros(F)
        self.params[cur_W_2] = np.random.randn(F, F, wH, wW ) * weight_scale
        self.params[cur_b_2] = np.zeros(F)
        
        # batchnorm parameters
        cur_gam_1 = ('Cgam%d1' % i)
        cur_gam_2 = ('Cgam%d2' % i)
        cur_bet_1 = ('Cbet%d1' % i)
        cur_bet_2 = ('Cbet%d2' % i)
        self.params[cur_gam_1] = np.ones(F)
        self.params[cur_gam_2] = np.ones(F)
        self.params[cur_bet_1] = np.zeros(F)
        self.params[cur_bet_2] = np.zeros(F)
    
    self.C1_bn_pars = [{'mode': 'train', 'use':use_batchnorm} for i in xrange(CN)]
    self.C2_bn_pars = [{'mode': 'train', 'use':use_batchnorm} for i in xrange(CN)]
    
    #################### [affine-relu]xAN weights and biases ####################
    
    # first affine relu layer
    aff_inp_size = F * H/(2**CN) * W/(2**CN)
    self.params['AW1'] = np.random.randn(aff_inp_size, hidden_dim ) * weight_scale
    self.params['Ab1'] = np.zeros(hidden_dim)
    self.params['Agam1'] = np.ones(hidden_dim)
    self.params['Abet1'] = np.zeros(hidden_dim)
    
    # other affine relu layers, if any
    for i in xrange(2, AN+1):
        cur_W = ('AW%d' % i)
        cur_b = ('Ab%d' % i)
        
        self.params[cur_W] = np.random.randn(hidden_dim, hidden_dim ) * weight_scale
        self.params[cur_b] = np.zeros(hidden_dim)
        
        # batchnorm parameters
        cur_gam = ('Agam%d' % i)
        cur_bet = ('Abet%d' % i)
        self.params[cur_gam] = np.ones(hidden_dim)
        self.params[cur_bet] = np.zeros(hidden_dim)
    
    self.A_bn_pars = [{'mode': 'train', 'use':use_batchnorm} for i in xrange(AN)]
    
    #################### output affine layer weights and biases ####################
    cur_W = ('AW%d' % (AN+1))
    cur_b = ('Ab%d' % (AN+1))
    
    self.params[cur_W] = np.random.randn(hidden_dim, num_classes ) * weight_scale
    self.params[cur_b] = np.zeros(num_classes)

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
    
    
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    ########################################################
    #################### initialisation ####################
    
    # retrieve mode
    mode = 'test' if y is None else 'train'
    
    # retrieve model sizes for for-loops
    CN, AN      = self.num_conv, self.num_hid
    
    # pass conv_param to the forward pass for the convolutional layer
    conv_par  = {'stride': 1, 'pad': (self.params['CW11'].shape[2] - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_par  = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    
    # set mode for batchnormalisation
    for bn_param in self.A_bn_pars:
        bn_param[mode] = mode
    
    scores = None # initialise scores
    caches = {} # create dictionary to store caches
    cur_X  = X # initialise first input
    
    ######################################################
    #################### forward pass ####################
    
    #################### [conv-relu-conv-relu-pool]xCN combinations ####################
    for i in xrange(1, CN+1):
        
        cur_W_1   = self.params[('CW%d1' % i)]
        cur_W_2   = self.params[('CW%d2' % i)]
        cur_b_1   = self.params[('Cb%d1' % i)]
        cur_b_2   = self.params[('Cb%d2' % i)]
        cur_gam_1 = self.params[('Cgam%d1' % i)]
        cur_gam_2 = self.params[('Cgam%d2' % i)]
        cur_bet_1 = self.params[('Cbet%d1' % i)]
        cur_bet_2 = self.params[('Cbet%d2' % i)]
        cur_bn_par_1 = self.C1_bn_pars[i-1]
        cur_bn_par_2 = self.C2_bn_pars[i-1]
        cur_cac   = ('CC%d'  % i)
        
        cur_X_1, C_cache_1 = conv_batch_relu_forward(cur_X, cur_W_1, cur_b_1, cur_gam_2, cur_bet_1, conv_par, cur_bn_par_1)
        #cur_X_1, C_cache_1 = conv_relu_forward(cur_X, cur_W_1, cur_b_1, conv_par)
        #cur_X_2, C_cache_2 = conv_relu_pool_forward(cur_X_1, cur_W_2, cur_b_2, conv_par, pool_par)
        cur_X_2, C_cache_2 = conv_batch_relu_pool_forward(cur_X_1, cur_W_2, cur_b_2, cur_gam_2, cur_bet_2, conv_par, cur_bn_par_2, pool_par)
        
        cur_X           = cur_X_2
        caches[cur_cac] = [C_cache_1, C_cache_2]
    
    
    #################### [affine-relu]x(AN-1) combinations ####################
    for i in xrange(1, AN+1):
        
        cur_W   = self.params[('AW%d' % i)]
        cur_b   = self.params[('Ab%d' % i)]
        cur_gam = self.params[('Agam%d' % i)]
        cur_bet = self.params[('Abet%d' % i)]
        cur_bn_par = self.A_bn_pars[i-1]
        cur_cac = ('AC%d' % i)
        
        #cur_X_A , A_cache  = affine_relu_forward(cur_X, cur_W, cur_b)
        cur_X_A, A_cache = affine_batch_relu_forward(cur_X, cur_W, cur_b, cur_gam, cur_bet, cur_bn_par)
        
        cur_X           = cur_X_A
        caches[cur_cac] = A_cache
        
    #################### output affine layer ####################
    out_W   = self.params[('AW%d' % (AN+1))]
    out_b   = self.params[('Ab%d' % (AN+1))]
    
    scores , out_cache  = affine_forward(cur_X, out_W, out_b)
    
    
    # return if test mode (y=None)
    if y is None:
      return scores
    
    #######################################################
    #################### backward pass ####################
    
    # backward pass initialisation
    loss, grads = 0, {}
    reg         = self.reg
    reg_loss    = 0.0
    
    # softmax loss and d_scores
    sof_loss, d_sco = softmax_loss(scores, y)
    
    #################### output affine later's derivatives ####################
    reg_out_W  = reg * out_W
    reg_loss  += 0.5 * np.sum(np.multiply(reg_out_W, out_W))
    
    d_cur_X, d_out_W, d_out_b = affine_backward(d_sco, out_cache)
   
    grads[('AW%d' % (AN+1))]      = d_out_W + reg_out_W
    grads[('Ab%d' % (AN+1))]      = d_out_b
        
    #################### [affine-relu]x(AN-1) combinations' derivatives ####################
    for i in xrange(AN, 0, -1):
        
        # regularization
        reg_cur_W  = reg * self.params[('AW%d' % i)]
        reg_loss  += 0.5 * np.sum(np.multiply(reg_cur_W, self.params[('AW%d' % i)]))
        
        # derivatives
        cur_cac = caches[('AC%d' % i)]
        #d_cur_X_A, d_cur_W, d_cur_b = affine_relu_backward(d_cur_X, cur_cac)
        d_cur_X_A, d_cur_W, d_cur_b, d_cur_gam, d_cur_bet = affine_batch_relu_backward(d_cur_X, cur_cac)
        d_cur_X = d_cur_X_A
        
        # update grads dictionary
        grads[('AW%d' % i)]         = d_cur_W + reg_cur_W
        grads[('Ab%d' % i)]         = d_cur_b
        grads[('Agam%d' % i)]       = d_cur_gam
        grads[('Abet%d' % i)]       = d_cur_bet
        
    
    #################### [conv-relu-conv-relu-pool]xCN combinations' derivatives ####################
    for i in xrange(CN, 0, -1):
        
        # regularization
        reg_cur_W_2  = reg * self.params[('CW%d2' % i)]
        reg_loss    += 0.5 * np.sum(np.multiply(reg_cur_W_2, self.params[('CW%d2' % i)]))
        reg_cur_W_1  = reg * self.params[('CW%d1' % i)]
        reg_loss    += 0.5 * np.sum(np.multiply(reg_cur_W_1, self.params[('CW%d1' % i)]))
        
        # derivatives
        C_cache_1, C_cache_2 = caches[('CC%d' % i)]
        
        d_cur_X_2, d_cur_W_2, d_cur_b_2, d_cur_gam_2, d_cur_bet_2 = conv_batch_relu_pool_backward(d_cur_X, C_cache_2)
        #d_cur_X_2, d_cur_W_2, d_cur_b_2 = conv_relu_pool_backward(d_cur_X, C_cache_2)
        #d_cur_X_1, d_cur_W_1, d_cur_b_1 = conv_relu_backward(d_cur_X_2, C_cache_1)
        d_cur_X_1, d_cur_W_1, d_cur_b_1, d_cur_gam_1, d_cur_bet_1 = conv_batch_relu_backward(d_cur_X_2, C_cache_1)
        
        d_cur_X              = d_cur_X_1
        
        # update grads dictionary
        grads[('CW%d2' % i)]        = d_cur_W_2 + reg_cur_W_2
        grads[('Cb%d2' % i)]        = d_cur_b_2
        grads[('CW%d1' % i)]        = d_cur_W_1 + reg_cur_W_1
        grads[('Cb%d1' % i)]        = d_cur_b_1
        grads[('Cgam%d2' % i)]      = d_cur_gam_2
        grads[('Cbet%d2' % i)]      = d_cur_bet_2
        grads[('Cgam%d1' % i)]      = d_cur_gam_1
        grads[('Cbet%d1' % i)]      = d_cur_bet_1
        
    # overall loss
    loss            = sof_loss + reg_loss  
    return loss, grads

pass
