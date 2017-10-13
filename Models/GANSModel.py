import theano
import theano.tensor as T
from Layers.LayerHelper import *
from Layers.Net import *

class GANSModel:
    def __init__(self,
                 _input):
        # ===== Create tensor variables to store input / output data =====
        self.X = _input

        # ===== Create model =====
        # ----- Create net -----
        self.net = ConvNeuralNet()
        self.net.net_name = 'GANs For Reconstruction'

        # ----- Retrieve info from net -----
        self.batch_size = self.X.shape[0]

        # ----- Input -----
        self.net.layer['input_2d'] = InputLayer(self.net, self.X)

        # ----- Stack 1 -----
        # --- Fully Connected 1 ---
        self.net.layer_opts['hidden_input_size']  = 1024
        self.net.layer_opts['hidden_output_size'] = 7 * 7 * 64
        self.net.layer_opts['hidden_WName'] = 'fc1_W'
        self.net.layer_opts['hidden_bName'] = 'fc1_b'
        self.net.layer['fc1'] = HiddenLayer(self.net, self.net.layer['input_2d'].output)

        # --- Reshape ---
        self.net.layer_opts['reshape_new_shape'] = (self.batch_size, 64, 7, 7)
        self.net.layer['fc1_re'] = ReshapeLayer(self.net, self.net.layer['fc1'].output)

        # ----- Stack 2 -----
        # --- Unpool ---
        self.net.layer_opts['unpool_filter_size'] = (2, 2)
        self.net.layer['unpool1']                 = Unpool2DLayer(self.net, self.net.layer['fc1_re'].output)

        # --- Transposed Convolution ---
        self.net.layer_opts['conv2D_filter_shape'] = (32, 64, 5, 5)
        self.net.layer_opts['conv2D_stride']       = (1, 1)
        self.net.layer_opts['conv2D_border_mode']  = (2, 2)
        self.net.layer_opts['conv2D_WName']        = 'conv1_W'
        self.net.layer_opts['conv2D_bName']        = 'conv1_b'
        self.net.layer['conv1'] = ConvLayer(self.net, self.net.layer['unpool1'].output)

        # ----- Stack 3 -----
        # --- Unpool ---
        self.net.layer_opts['unpool_filter_size'] = (2, 2)
        self.net.layer['unpool2']                 = Unpool2DLayer(self.net, self.net.layer['conv1'].output)

        # --- Transposed Convolution ---
        self.net.layer_opts['conv2D_filter_shape'] = (1, 32, 5, 5)
        self.net.layer_opts['conv2D_stride']       = (1, 1)
        self.net.layer_opts['conv2D_border_mode']  = (2, 2)
        self.net.layer_opts['conv2D_WName'] = 'conv2_W'
        self.net.layer_opts['conv2D_bName'] = 'conv2_b'
        self.net.layer['conv2'] = ConvLayer(self.net, self.net.layer['unpool2'].output)

        self.params = self.net.layer['fc1'].params + \
                      self.net.layer['conv1'].params + \
                      self.net.layer['conv2'].params