import theano
import theano.tensor as T
from Layers.LayerHelper import *
from Layers.Net import *

class CNNModel:
    def __init__(self,
                 _input,
                 _params):
        # ===== Create tensor variables to store input / output data =====
        self.X = _input

        # ===== Create model =====
        # ----- Create net -----
        self.net = ConvNeuralNet()
        self.net.net_name = 'CNN for Feature Extraction'

        # ----- Retrieve info from net -----
        self.batch_size = self.X.shape[0]

        # ----- Input -----
        self.net.layer['input_4d'] = InputLayer(self.net, self.X)

        # ----- Stack 1 -----
        # --- Conv 1 ---
        self.net.layer_opts['conv2D_filter_shape'] = (32, 1, 5, 5)
        self.net.layer_opts['conv2D_stride']       = (1, 1)
        self.net.layer_opts['conv2D_border_mode']  = (2, 2)
        self.net.layer_opts['conv2D_W']            = _params[0]
        self.net.layer_opts['conv2D_WName']        = 'conv1_W'
        self.net.layer_opts['conv2D_b']            = _params[1]
        self.net.layer_opts['conv2D_bName']        = 'conv1_b'
        self.net.layer['conv1'] = ConvLayer(self.net, self.net.layer['input_4d'].output)

        # --- Relu 1 ---
        self.net.layer['relu1'] = ReLULayer(self.net.layer['conv1'].output)

        # --- Pool 1 ---
        self.net.layer_opts['pool_mode'] = 'max'
        self.net.layer['pool1']          = Pool2DLayer(self.net, self.net.layer['relu1'].output)

        # ----- Stack 2 -----
        # --- Conv 2 ---
        self.net.layer_opts['conv2D_filter_shape'] = (64, 32, 5, 5)
        self.net.layer_opts['conv2D_stride']       = (1, 1)
        self.net.layer_opts['conv2D_border_mode']  = (2, 2)
        self.net.layer_opts['conv2D_W']            = _params[2]
        self.net.layer_opts['conv2D_WName']        = 'conv2_W'
        self.net.layer_opts['conv2D_b']            = _params[3]
        self.net.layer_opts['conv2D_bName']        = 'conv2_b'
        self.net.layer['conv2'] = ConvLayer(self.net, self.net.layer['pool1'].output)

        # --- Relu 2 ---
        self.net.layer['relu2'] = ReLULayer(self.net.layer['conv2'].output)

        # --- Pool 1 ---
        self.net.layer_opts['pool_mode'] = 'max'
        self.net.layer['pool2']          = Pool2DLayer(self.net, self.net.layer['relu2'].output)

        # ----- Stack 3 -----
        # --- Reshape ---
        self.net.layer_opts['reshape_new_shape'] = (self.batch_size, 7 * 7 * 64)
        self.net.layer['pool2_re'] = ReshapeLayer(self.net, self.net.layer['pool2'].output)

        # --- Fully Connected Layer ---
        self.net.layer_opts['hidden_input_size']  = 7 * 7 * 64
        self.net.layer_opts['hidden_output_size'] = 1024
        self.net.layer_opts['hidden_W']           = _params[4]
        self.net.layer_opts['hidden_WName']       = 'fc1_W'
        self.net.layer_opts['hidden_b']           = _params[5]
        self.net.layer_opts['hidden_bName']       = 'fc1_b'
        self.net.layer['fc1'] = HiddenLayer(self.net, self.net.layer['pool2_re'].output)

        # ----- Stack 4 -----
        # --- Dropout ---
        self.net.layer_opts['drop_rate']  = 0.5
        self.net.layer_opts['drop_shape'] = self.net.layer['fc1'].output.shape
        self.net.layer['fc1_drop']        = DropoutLayer(self.net, self.net.layer['fc1'].output)

        # --- Fully Connected Layer ---
        self.net.layer_opts['hidden_input_size']  = 1024
        self.net.layer_opts['hidden_output_size'] = 10
        self.net.layer_opts['hidden_W']           = _params[6]
        self.net.layer_opts['hidden_WName']       = 'fc2_W'
        self.net.layer_opts['hidden_b']           = _params[7]
        self.net.layer_opts['hidden_bName']       = 'fc2_b'
        self.net.layer['fc2'] = HiddenLayer(self.net, self.net.layer['fc1_drop'].output)

        # --- Softmax Layer ---
        self.net.layer['prob'] = SoftmaxLayer(self.net, self.net.layer['fc2'].output)

        # --- Params ---
        self.params = self.net.layer['conv1'].params + \
                      self.net.layer['conv2'].params + \
                      self.net.layer['fc1'].params + \
                      self.net.layer['fc2'].params

    def save_model(self, file):
        [pickle.dump(param.get_value(borrow = True), file, 2) for param in self.params]

    def load_model(self, file):
        [param.set_value(cPickle.load(file), borrow = True) for param in self.params]