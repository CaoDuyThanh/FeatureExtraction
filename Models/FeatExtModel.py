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
        self.batch_size = self.Y.shape[0]

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
        _gens         = self.gans_net.net.layer['sig1'].output
        _gens_params  = self.gans_net.params

        # ----- Feature Extraction Model -----
        self.pred_net = CNNModel(_input  = _gens,
                                 _params = _feature_params)
        _pred_probs   = self.pred_net.net.layer['prob'].output


        # ===== Cost function =====
        # ----- Featext Cost -----
        _true_feat  = _feature_probs[ : self.batch_size, ]
        _noise_feat = _feature_probs[-self.batch_size:, ]
        _feature_cost  = (-T.mean(T.log(_true_feat[T.arange(self.batch_size), self.Y])) \
                          -T.mean(T.log(1 - _noise_feat[T.arange(self.batch_size), self.Y]))) / 2
        _feature_grads =  T.grad(_feature_cost, _feature_params)

        # ----- Optimizer -----
        _feature_optimizer = AdamGDUpdate(self.featext_net.net, params = _feature_params, grads = _feature_grads)
        self.feature_optimizer = _feature_optimizer

        # ----- Gens Cost -----
        _gens_cost  = -T.mean(T.log(_pred_probs[T.arange(self.batch_size), self.Y]))
        _gens_grads =  T.grad(_gens_cost, _gens_params)

        # ----- Optimizer -----
        _gens_optimizer = AdamGDUpdate(self.featext_net.net, params = _gens_params, grads = _gens_grads)
        self.gens_optimizer = _gens_optimizer

        # ===== Function =====
        # ----- Featext function -----
        self.feat_train_func = theano.function(inputs  = [self.featext_net.net.layer['fc1_drop'].state,
                                                          _feature_optimizer.learning_rate,
                                                          self.X,
                                                          self.Y],
                                               updates = _feature_optimizer.updates,
                                               outputs = [_feature_cost])

        self.feat_ext_func = theano.function(inputs  = [self.X],
                                             outputs = [_feature])

        # ----- Gens function -----
        self.gens_train_func = theano.function(inputs  = [self.pred_net.net.layer['fc1_drop'].state,
                                                          _gens_optimizer.learning_rate,
                                                          self.X,
                                                          self.Y],
                                               updates = _gens_optimizer.updates,
                                               outputs = [_gens_cost])

        self.gens_gen_img_func1 = theano.function(inputs  = [self.X],
                                                  outputs = [_gens])

        self.gens_gen_img_func = theano.function(inputs  = [self.X],
                                                 outputs = [_gens.dimshuffle([0, 2, 3, 1])])


    def load_model(self,
                   _file):
        self.featext_net.load_model(_file)
        self.gans_net.load_model(_file)

    def save_model(self,
                   _file):
        self.featext_net.save_model(_file)
        self.gans_net.save_model(_file)

    def load_state(self,
                   _file):
        self.featext_net.load_model(_file)
        self.gans_net.load_model(_file)
        self.feature_optimizer.load_model(_file)
        self.gens_optimizer.load_model(_file)

    def save_state(self,
                   _file):
        self.featext_net.save_model(_file)
        self.gans_net.save_model(_file)
        self.feature_optimizer.save_model(_file)
        self.gens_optimizer.save_model(_file)


