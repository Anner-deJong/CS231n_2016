import numpy as np

class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      for j in xrange(num_train):
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################
        
        dists[i,j] = np.sqrt(np.sum(np.square(X[i]-self.X_train[j])))
        
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################
      
      diff_matr = np.subtract(self.X_train,X[i])
      dists[i] = np.sqrt(np.sum(np.square(diff_matr),1))
      
      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    
    # L2 difference between test_i and train_j is calculated as the square root of the sum of:
    # (1) sum of the element-wise squares of test_i
    # (2) sum of the element-wise squares of train_j
    # (3) minus 2 x sum of the element-wise multiplication of test_i and train_j
    
    # creating i x j matrix with (1) + (2) entries:
    # transpose if necessary to get the correct size
    sum_sq_test  = np.transpose(np.array([np.sum(np.square(X),1)]))
    sum_sq_train = np.array([np.sum(np.square(self.X_train),1)])
    
    # broadcast these arrays into a num_test (i) x num_train matrix (j) sum
    [broadc_test, broadc_train] = np.broadcast_arrays(sum_sq_test,sum_sq_train)
    sq_sum_mat = np.add(broadc_test, broadc_train)
    
    # creating i x j matrix with (3) entry
    mult_sum_mat = np.multiply(np.dot(-X,self.X_train.transpose()),2)
    
    # final step: sum (1) - (3) matrices and take the element-wise square root
    dists = np.sqrt(np.add(sq_sum_mat, mult_sum_mat))
    
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    #import time
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in xrange(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # testing point, and use self.y_train to find the labels of these       #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################
        
          #########################################################################
          #          Less efficient method using apply_along_axis:                #
          #########################################################################

          #tic = time.time()
          #
          #closest_y = self.y_train[np.argsort(dists,axis=1)[:,0:k]]
          #y_pred = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), 1, closest_y)
          #
          #toc = time.time()
          #print 'apply_along_axis took %fs seconds' % (toc - tic)
          
          #########################################################################
          #       Less efficient method using for loop and non for loop           #
          #########################################################################  
          
          #tic = time.time()
          #
          #closest_y = self.y_train[np.argsort(dists,axis=1)[:,0:k]]
          #for i in xrange(num_test):
          #    y_pred[i] = np.argmax(np.bincount(closest_y[i,:]))
          #
          #toc = time.time()
          #print 'combined took %fs seconds' % (toc - tic)  
            
      closest_y = self.y_train[np.argsort(dists[i,:])[0:k]]  
            
      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
        
      y_pred[i] = np.argmax(np.bincount(closest_y))
 
      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################

    return y_pred

