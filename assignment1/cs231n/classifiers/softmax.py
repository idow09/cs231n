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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in xrange(num_train):
        soft_scores = softmax(X[i].dot(W))
        loss += -np.log(soft_scores[y[i]])
        dW[:, y[i]] += -X[i] * (1 - soft_scores[y[i]])
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            dW[:, j] += X[i] * soft_scores[j]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W

    return loss, dW


def softmax(s):
    exps = np.exp(s - np.max(s))
    return exps / np.sum(exps)


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    num_train = X.shape[0]

    all_rows = np.arange(num_train)
    soft_scores = softmax2d(X.dot(W))
    loss = np.sum(-np.log(soft_scores[all_rows, y]))

    soft_scores[all_rows, y] -= 1
    dW = X.T.dot(soft_scores)

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W

    return loss, dW


def softmax2d(s):
    exps = np.exp(s - np.max(s))
    return exps / np.sum(exps, axis=1).reshape(-1, 1)
