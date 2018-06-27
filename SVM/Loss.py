import numpy as np

def L_i(x, y, W):
  """
  unvectorized version. Compute the multiclass svm loss for a single example (x,y)
  - x is a column vector representing an image (e.g. 3073 x 1 in CIFAR-10)
    with an appended bias dimension in the 3073-rd position (i.e. bias trick)
  - y is an integer giving index of correct class (e.g. between 0 and 9 in CIFAR-10)
  - W is the weight matrix (e.g. 10 x 3073 in CIFAR-10)
  """
  delta = 1.0 # see notes about delta later in this section
  scores = W.dot(x) # scores becomes of size 10 x 1, the scores for each class
  correct_class_score = scores[y]
  D = W.shape[0] # number of classes, e.g. 10
  loss_i = 0.0
  for j in xrange(D): # iterate over all wrong classes
    if j == y:
      # skip for the true class to only loop over incorrect classes
      continue
    # accumulate loss for the i-th example
    loss_i += max(0, scores[j] - correct_class_score + delta)
  return loss_i

def L_i_vectorized(x, y, W):
  """
  A faster half-vectorized implementation. half-vectorized
  refers to the fact that for a single example the implementation contains
  no for loops, but there is still one loop over the examples (outside this function)
  """
  delta = 1.0
  scores = W.dot(x)
  margins = np.maximum(0, scores - scores[y] + delta)
  margins[y] = 0
  loss_i = np.sum(margins)
  return loss_i

def mf(v, i, W):
    return L_i(v, i, W)

vfunc = np.vectorize(mf)

def L(X, y, W):
  """
  fully-vectorized implementation :
  - X holds all the training examples as columns (e.g. 3073 x 50,000 in CIFAR-10)
  - y is array of integers specifying correct class (e.g. 50,000-D array)
  - W are weights (e.g. 10 x 3073)
  """
  delta = 1.0
  matrix_of_scores = W.dot(X).T
  s_yi = matrix_of_scores[np.arange(y.shape[0]), y]
  matrix_of_loss = matrix_of_scores - s_yi.reshape((y.shape[0], 1)) + delta
  matrix_of_loss[np.arange(y.shape[0]), y] = 0
  matrix_of_loss[matrix_of_loss < 0] = 0
  return matrix_of_loss.sum()/matrix_of_loss.shape[0]



if __name__ == "__main__":
    W = np.array([[5,6,7,3], [2,4,5,1], [7,3,1,5]])
    y = np.array([1,0,2,2])
    X = np.array([[1,3,4,8], [5,6,3,0], [9,3,1,9], [7,8,4,2]])

    print(L(X, y, W))
