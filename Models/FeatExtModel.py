import theano
import theano.tensor as T
import numpy
import cv2
from Layers.LayerHelper import *
from Layers.Net import *
from Models.CNNModel import *
from Models.GANSModel import *

class FEAEXTModel():
    def __init__(self):
        # ===== Create tensor variables to store input / output data =====
        self.X = T.tensor4('X')
        self.Y = T.ivector('Y')

        # ===== Extract Info =====
        self.batch_size = self.X.shape[0]

        # ===== Create model =====
        # ----- Feature Extraction Model -----
        self.featext_net = CNNModel(_input  = self.X,
                                    _params = [None, None, None, None,
                                               None, None, None, None])
        _feature         = self.featext_net.net.layer['fc1'].output
        _feature_params  = self.featext_net.params
        _feature_probs   = self.featext_net.net.layer['prob'].output
        _feature_preds   = T.argmax(_feature_probs, axis = 1)

        # ----- GANs Model -----
        self.gans_net = GANSModel(_input = _feature)
        _gens         = self.gans_net.net.layer['conv2'].output
        _gens_params  = self.gans_net.params

        # ----- Feature Extraction Model -----
        self.pred_net = CNNModel(_input  = _gens,
                                 _params = _gens_params)
        _pred_probs   = self.pred_net.net.layer['prob'].output


        # ===== Cost function =====
        # ----- Featext Cost -----
        _feature_cost  = T.mean(T.log(_feature_probs[T.arange(1, self.batch_size), self.Y]))
        _feature_grads = T.grad(_feature_cost, _feature_params)

        # ----- Optimizer -----
        _feature_optimizer = AdamGDUpdate(self.featext_net, params = _feature_params, grads = _feature_grads)

        # ----- Gens Cost -----
        _gens_cost  = T.mean(T.log(1 - _pred_probs[T.arange(1, self.batch_size), self.Y]))
        _gens_grads = T.grad(_gens_cost, _gens_params)

        # ----- Optimizer -----
        _gens_optimizer = AdamGDUpdate(self.featext_net, params = _gens_params, grads = _gens_grads)


        # ===== Function =====
        # ----- Featext function -----
        self.feat_train_func = theano.function(inputs  = [self.featext_net.net.layer['fc1_drop'].state,
                                                          _feature_optimizer.learning_rate,
                                                          self.X,
                                                          self.Y],
                                               updates = _feature_optimizer.updates,
                                               outputs = [_feature_cost])

        # ----- Gens function -----
        self.gens_train_func = theano.function(inputs  = [self.featext_net.net.layer['fc1_drop'].state,
                                                          _gens_optimizer.learning_rate,
                                                          self.X,
                                                          self.Y],
                                               updates = _gens_optimizer.updates,
                                               outputs = [_gens_cost])


    def create_func(self,
                    _layer_name):
        return theano.function(inputs  = [self.X],
                               outputs = [self.net.layer[_layer_name].output])

    def load_caffe_model(self,
                         _caffe_prototxt_path,
                         _caffe_model_path):
        self.net.load_caffe_model(_caffe_prototxt_path, _caffe_model_path)

    def load_batch_norm(self,
                        _file):
        self.net.layer['conv4_3_norm_batch_norm'].load_model(_file)
        self.net.layer['relu7_batch_norm'].load_model(_file)
        self.net.layer['conv6_2_relu_batch_norm'].load_model(_file)
        self.net.layer['conv7_2_relu_batch_norm'].load_model(_file)
        self.net.layer['conv8_2_relu_batch_norm'].load_model(_file)
        self.net.layer['pool6_batch_norm'].load_model(_file)

    def test_network(self,
                     _im):
        return self.feat_func(_im)[0][0], self.pred_func(_im)
