from builtins import object

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
        - filter_size: Width/height of filters to use in the convolutional layer
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

        C, H, W = input_dim
        self.params['W1'] = np.random.normal(scale=weight_scale, size=(num_filters, C, filter_size, filter_size))
        self.params['b1'] = np.zeros(num_filters)
        self.params['W2'] = np.random.normal(scale=weight_scale, size=(num_filters * H * W // 4, hidden_dim))
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
        self.params['b3'] = np.zeros(num_classes)

        for k, v in self.params.items():
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
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        # Forward
        conv_out, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        hidden_out, cache2 = affine_relu_forward(conv_out, W2, b2)
        scores, cache3 = affine_forward(hidden_out, W3, b3)
        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores
        loss, dout = softmax_loss(scores, y)

        # Regularization forward
        loss += 0.5 * self.reg * np.sum(W1 * W1)
        loss += 0.5 * self.reg * np.sum(W2 * W2)
        loss += 0.5 * self.reg * np.sum(W3 * W3)

        # Backward
        grads = {}
        dhidden, dW3, db3 = affine_backward(dout, cache3)
        grads['W3'] = dW3
        grads['b3'] = db3
        dconv, dW2, db2 = affine_relu_backward(dhidden, cache2)
        grads['W2'] = dW2
        grads['b2'] = db2
        _, dW1, db1 = conv_relu_pool_backward(dconv, cache1)
        grads['W1'] = dW1
        grads['b1'] = db1

        # Regularization backward
        grads['W1'] += self.reg * self.params['W1']
        grads['W2'] += self.reg * self.params['W2']
        grads['W3'] += self.reg * self.params['W3']

        return loss, grads
