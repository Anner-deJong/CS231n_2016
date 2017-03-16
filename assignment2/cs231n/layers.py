import numpy as np
from cs231n.im2col import *

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  
  # define shapes
  N = x.shape[0]
  D = np.prod(x.shape[1:])
  M = w.shape[1]
  
  # shape x to example . features per sample matrix
  x_arr = x.reshape(N, D)

  # calculate output including bias
  out = x_arr.dot(w) + b
    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)
    - b: Biases, of shape (M,)
    
  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  
  # define shapes
  N = x.shape[0]
  D = np.prod(x.shape[1:])
  
  # shape x to example . features per sample matrix
  x_arr = x.reshape(N, D)

  # calculate partial derivatives
  dx_arr = dout.dot(w.transpose())
  dx     = dx_arr.reshape(N, *x.shape[1:])
  dw     = x_arr.transpose().dot(dout)
  db     = dout.sum(axis=0)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################

  out = np.maximum(0,x)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  
  dx = np.where(x > 0, dout, 0)
    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
       
    #sample_mean  = np.mean( x, axis=0)
    #sample_var   = np.var(  x, axis=0)
    
    x_mean       = np.sum( x, axis = 0) / N
    
    x_susq       = np.sum( np.power(x, 2), axis = 0)
    x_sqsu       = np.power( np.sum(x, axis = 0), 2)
    x_var        = x_susq / N - x_sqsu / (N * N)
    x_std        = np.sqrt(x_var + eps)
    
    x_nor        = (x - x_mean) / x_std
    
    out          = gamma    * x_nor + beta
    
    running_mean = momentum * running_mean + (1 - momentum) * x_mean
    running_var  = momentum * running_var  + (1 - momentum) * x_var
    
    cache          = {}
    cache['gamma'] = gamma
    cache['x']     = x
    cache['x_std'] = x_std
    cache['x_nor'] = x_nor
    cache['eps']   = eps
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    
    x_mean       = (x - running_mean )
    x_std        = np.sqrt(running_var + eps)
    x_nor        = x_mean / x_std
    
    out          = gamma * x_nor + beta
    #out = gamma * ( (x - running_mean ) / np.sqrt(running_var + eps)) + beta
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  
  x            = cache['x']
  x_std        = cache['x_std']
  x_nor        = cache['x_nor']
  gamma        = cache['gamma']
  eps          = cache['eps']
  #x_std        = cache['x_std']
  N,D          = x.shape
    
  sample_mean  = np.mean( x, axis=0)
  #sample_var   = np.var(  x, axis=0)
  x_mean       = (x - sample_mean )
  #x_std        = np.sqrt(sample_var + eps)
  #x_nor        = x_mean / x_std
    
  v            = 1. / x_std

  dx_nor       = np.multiply(dout, gamma)
  dv           = np.sum( np.multiply(dx_nor, x_mean), axis=0)  
  
  dm_1         = dx_nor * v
  dm_2_sca     = -1. / N / x_std**(3.) * dv
  dm_2         = np.multiply( x_mean, dm_2_sca )

  dm = dm_1 + dm_2
  
  dx     = dm - (np.sum( dm, axis=0) / N)  
  dgamma = np.sum(np.multiply(dout, x_nor), axis=0)
  dbeta  = np.sum( dout, axis=0)
    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  
  x      = cache['x']
  x_sig  = cache['x_std']
  x_nor  = cache['x_nor']
  x_mu   = np.mean( x, axis=0)
  N,D    = x.shape
  
  gamma  = cache['gamma']
  eps    = cache['eps']
  
  dx_nor = np.multiply(dout, gamma)

  dx     = (1./x_sig) * ( dx_nor  -  (np.sum(dx_nor, axis=0)/N)  -  (x - x_mu)* np.sum((x - x_mu)* dx_nor, axis=0)* (x_sig**(-2.) /N) )
  dgamma = np.sum(np.multiply(dout, x_nor), axis=0)
  dbeta  = np.sum( dout, axis=0)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    
    mask = np.random.rand(*x.shape) > p
    out  = np.multiply(mask, x) / (1 - p)
    
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    
    out = x
    
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    
    p  = dropout_param['p']
    dx = np.multiply(dout, mask) / (1 - p)
    
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width WW (HH?).

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, wH, wW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, oH, oW) where H' and W' are given by
    oH = 1 + (H + 2 * pad - wH) / stride
    oW = 1 + (W + 2 * pad - wW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  
  # key values
  N, C, H, W   = x.shape
  F, C, wH, wW = w.shape

  pad = conv_param['pad']
  str = conv_param['stride']
  
  pH = H + 2 * pad
  pW = W + 2 * pad

  oH = 1 + (pH - wH) / str
  oW = 1 + (pW - wW) / str
  
  
  # padding    x_p
  x_p = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
  
  
  # input column    x_col (N, C, H, W)   -> (N, C*wH*wW, oH, oW)
  # retrieving correct indexes
  or2d_ind = (np.arange(wH) * pH)         [:, None] + np.arange(wW)
  or3d_ind = (np.arange(C)  * pH * pW)    [:, None] + or2d_ind.ravel()
  strides  = (np.arange(oH) * pH * str)   [:, None] + np.arange(oW) * str
  str_ind  = np.ravel(or3d_ind)           [:, None] + strides.ravel()
  sam_ind  = (str_ind)[None, :] + (np.arange(N)  * pH * pW * C)[:, None, None]  
  
  x_col    = np.take (x_p,  sam_ind)
  
  
  # weight column    w_col (F, C, wH, wW) -> (F, C*wH*wW)
  w_col = np.reshape(w, (F, C*wH*wW))
  

  # output column    out   (N, F, oH, oW)
  out_col  = np.einsum('ijk,lj->ilk',x_col, w_col) + b[None, :, None]
  out      = np.reshape(out_col, (N, F, oH, oW))
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, x_col, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  
  # forward pass cache
  x, x_col, w, b, conv_param = cache
  
  # key values
  N, C, H, W   = x.shape
  F, C, wH, wW = w.shape

  pad = conv_param['pad']
  str = conv_param['stride']
  
  pH = H + 2 * pad
  pW = W + 2 * pad

  oH = 1 + (pH - wH) / str
  oW = 1 + (pW - wW) / str
  
  
  # x gradients - for loop way 
  d_col     = np.reshape(dout, (N, F, oH*oW))
  w_col     = np.reshape(w, (F, C*wH*wW))
  dx_col    = np.sum( np.multiply(w_col[None, :, :, None], d_col[:, :, None, :]), 1)
  dx_cube   = np.reshape( dx_col, (N, C, wH, wW, oH*oW))
  
  dx_p      = np.zeros((N, C, H + 2*pad, W + 2*pad), dtype=dx_col.dtype)
  for xx in xrange(oW):
    for yy in xrange(oH):
        dx_p[:, :, yy*str:yy*str+wH, xx*str:xx*str+wW] += dx_cube[:, :, :, :, yy*oW + xx]
  
  dx = dx_p[:, :, pad:-pad, pad:-pad]
  
  
  # x gradients - col2im way
  #---------------------------------------------------------------------------# 
  #dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(F, -1)
  #dx_cols = w.reshape(F, -1).T.dot(dout_reshaped)
  #dx = col2im_indices(dx_cols, x.shape, wH, wW, pad, str)
  
  # x gradients - indexing way
  #---------------------------------------------------------------------------# 
  # retrieving correct indexes
  #C_ind    = np.tile(np.tile(np.arange(C) [:,None], wH*wW).ravel()[:, None], oH*oW)
  #
  #H_single = np.tile(np.tile(np.arange(wH)[:,None], wW).ravel(), C)
  #H_stride = np.tile((np.arange(oH) * str)[:,None], oW ).ravel()
  #H_ind    = H_single[:, None] + H_stride
  #
  #W_single = np.tile(np.tile(np.arange(wW),wH).ravel(), C)
  #W_stride = np.tile((np.arange(oW) * str), oH ).ravel()
  #W_ind    = W_single[:, None] + W_stride
  #
  #d_col     = np.reshape(dout, (N, F, oH*oW))
  #w_col     = np.reshape(w, (F, C*wH*wW))
  #dx_col    = np.sum( np.multiply(w_col[None, :, :, None], d_col[:, :, None, :]), 1)
  #
  #dx_p      = np.zeros((N, C, H + 2*pad, W + 2*pad), dtype=dx_col.dtype)
  #np.add.at(dx_p, (slice(None), C_ind, H_ind, W_ind), dx_col)
  #dx = dx_p[:, :, pad:-pad, pad:-pad]
  #---------------------------------------------------------------------------#
  
  
  # weight gradients
  dw_col    = np.multiply(x_col[:, None, :], d_col[:, :, None, :])
  dw_sum    = np.sum(dw_col, (0, 3))
  
  dw        = np.reshape(dw_sum, (F, C, wH, wW))
  
  
  # bias gradients
  db = np.sum(dout, axis=(0,2,3))
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  
  # key values
  N, C, H, W   = x.shape

  pH  = pool_param['pool_height']
  pW  = pool_param['pool_width']
  str = pool_param['stride']

  oH = 1 + (H - pH) / str
  oW = 1 + (W - pW) / str
  
  
  # retrieve correct indices
  or2d_ind  = (np.arange(pH) * H)        [:, None] + np.arange(pW)
  or3d_ind  = (np.arange(C)  * H * W)    [:, None] + or2d_ind.ravel()
  strides   = (np.arange(oH) * H * str)  [:, None] + np.arange(oW) * str
  str_ind   = or3d_ind                [:, :, None] + strides.ravel() [None, None, :] 
  sam_ind   = (str_ind)                  [None, :] + (np.arange(N) * H * W * C)[:, None, None, None]  
  
  
  # retrieve correct nodes, max over pH*pW, and reshape to oH, oW
  x_col     = np.reshape(np.transpose(np.take (x,  sam_ind), (0,1,3,2)), (N*C*oH*oW, pH*pW))
  x_max     = np.amax(x_col, 1)
  x_max_ind = np.argmax(x_col, 1)
  out       = np.reshape(x_max, (N, C, oH, oW))
  
  #out = im2col_indices(x, pH, pW, 0, str)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, x_max_ind, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  
  # forward pass cache
  x, x_max_ind, pool_param = cache
  
  # key values
  N, C, H, W   = x.shape
  
  pH  = pool_param['pool_height']
  pW  = pool_param['pool_width']
  str = pool_param['stride']
  
  oH = 1 + (H - pH) / str
  oW = 1 + (W - pW) / str
    
  # creating x_col size matrix with dout at max positions 
  dx_zer   = np.zeros((N*C*oH*oW, pH*pW))
  hulp_ind = np.arange(N*C*oH*oW)
  dx_zer[hulp_ind, x_max_ind] = dout.ravel()
  dx_col   = np.transpose(np.reshape(dx_zer, (N, C, oH*oW, pH*pW)),(0,1,3,2))
  
  dx_cube   = np.reshape( dx_col, (N, C, pH, pW, oH*oW))
  
  dx = np.zeros((x.shape))
  for xx in xrange(oW):
     for yy in xrange(oH):
        dx[:, :, yy*str:yy*str+pH, xx*str:xx*str+pW] += dx_cube[:, :, :, :, yy*oW + xx]
  
  #dx_col_temp = np.reshape(np.transpose(dx_col,(1,2,0,3)),(C*pH*pW, N*oH*oW))
  #dout_reshaped = dout.transpose(2, 3, 0, 1).flatten()
  #dx_cols = np.zeros_like(x_cols)
  #dx_cols[x_cols_argmax, np.arange(dx_cols.shape[1])] = dout_reshaped
  #dx = col2im_indices(dx_col_temp, (N*C,1, H, W), pH, pW, 0, str)
  #dx = np.reshape(dx, x.shape)
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, F, H, W)
  - gamma: Scale parameter, of shape (F,)
  - beta: Shift parameter, of shape (F,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, F, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  
  N, C, H, W       = x.shape
  x_shape          = np.reshape(np.transpose(x, (0,2,3,1)), (-1, C))
  out_shape, cache = batchnorm_forward(x_shape, gamma, beta, bn_param)
  out              = np.transpose(np.reshape(out_shape, (N, H, W, C)), (0,3,1,2))
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  
  N, C, H, W              = dout.shape
  dout_shape              = np.reshape(np.transpose(dout, (0,2,3,1)), (-1, C))
  dx_shape, dgamma, dbeta = batchnorm_backward_alt(dout_shape, cache)
  dx                      = np.transpose(np.reshape(dx_shape, (N, H, W, C)), (0,3,1,2))
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
