import theano.tensor as T
from UtilLayer import *
from BasicLayer import *

class BatchNormLayer(BasicLayer):
    def __init__(self,
                 _net,
                 _input):
        self.batch_normalize_shape = _net.layer_opts['batch_normalize_shape']
        self.scale_name            = _net.layer_opts['batch_normalize_scale_name']

        # Save all information to its layer
        self.mu  = create_shared_parameter(
                        _rng      = _net.net_opts['rng'],
                        _shape    = self.batch_normalize_shape,
                        _name_var = self.scale_name
                    )
        self.std = create_shared_parameter(
                        _rng      = _net.net_opts['rng'],
                        _shape    = self.batch_normalize_shape,
                        _name_var = self.scale_name
                    )

        self.params = [self.mu, self.std]

        _output_shape  = T.shape(_input)
        _mu = self.mu.reshape((1, self.batch_normalize_shape[0], 1, 1))
        _mu = T.extra_ops.repeat(
            _mu,
            _output_shape[0],
            0)
        _mu = T.extra_ops.repeat(
            _mu,
            _output_shape[2],
            2)
        _mu = T.extra_ops.repeat(
            _mu,
            _output_shape[3],
            3)

        _std = self.std.reshape((1, self.batch_normalize_shape[0], 1, 1))
        _std = T.extra_ops.repeat(
            _std,
            _output_shape[0],
            0)
        _std = T.extra_ops.repeat(
            _std,
            _output_shape[2],
            2)
        _std = T.extra_ops.repeat(
            _std,
            _output_shape[3],
            3)
        self.output = (_input - _mu) / _std