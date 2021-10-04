from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras import backend as K



class PreInput:
    def __init__(self, inputSize, C):
        self.C = C

        def mean_normalizeation(tensor):
            return tensor - self.C.meanBGR

        def swap_channels(tensor):
            return K.stack([tensor[..., 2], tensor[..., 1], tensor[..., 0]], axis=-1)

        x_input = Input(inputSize, name='input')
        x = Lambda(mean_normalizeation, name='mean_normalization')(x_input)
        x = Lambda(swap_channels, name='swap_channels')(x)

        self.x_input = x_input
        self.x = x