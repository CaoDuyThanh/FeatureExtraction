import theano.tensor as T

class Unpool2DLayer():
    def __init__(self,
                 _net,
                 _input):
        # Save config information to its layer
        self.Ws            = _net.layer_opts['unpool_filter_size']
        self.new_shape     = self.get_output_shape_for(_input.shape, self.Ws)

        upscaled    = T.zeros(shape = self.new_shape,
                              dtype = _input.dtype)
        self.output = T.set_subtensor(upscaled[:, :, ::self.Ws[2], ::self.Ws[3]], _input)

    def get_output_shape_for(self, _input_shape, _scale):
        output_shape = list(_input_shape)  # copy / convert to mutable list
        if output_shape[0] is not None:
            output_shape[0] *= _scale[0]
        if output_shape[1] is not None:
            output_shape[1] *= _scale[1]
        if output_shape[2] is not None:
            output_shape[2] *= _scale[2]
        if output_shape[3] is not None:
            output_shape[3] *= _scale[3]

        return tuple(output_shape)