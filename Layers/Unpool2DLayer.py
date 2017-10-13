import theano.tensor as T

class Unpool2DLayer():
    def __init__(self,
                 _net,
                 _input):
        # Save config information to its layer
        self.Ws            = _net.layer_opts['unpool_filter_size']

        upscaled    = T.zeros(shape = _input.shape,
                              dtype = _input.dtype)
        self.output = T.set_subtensor(upscaled[:, :, ::self.Ws[0], ::self.Ws[1]], _input)