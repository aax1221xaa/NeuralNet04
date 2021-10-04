from tensorflow.keras import layers
from tensorflow.keras.initializers import Constant
from tensorflow.keras import backend as K

from source.mic import *


class L2Norm(layers.Layer):
    def __init__(self, gamma_init, **kwargs):
        super(L2Norm, self).__init__(**kwargs)
        self.gamma_init = gamma_init
        self.axis = 3

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name= 'L2_gamma',
            shape= [input_shape[self.axis]],
            dtype= 'float32',
            initializer= Constant(self.gamma_init),
            trainable= True
        )
        super(L2Norm, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return self.gamma * K.l2_normalize(inputs, self.axis)


    def get_config(self):
        config = super(L2Norm, self).get_config()
        config.update({
            'gamma_init': self.gamma_init
        })

        return config

class PriorBox(layers.Layer):
    def __init__(self, img_size, base_anchor, **kwargs):
        self.img_width = img_size
        self.img_height = img_size
        self.base_anchor = base_anchor
        self.n_base_anchor = len(base_anchor)
        super(PriorBox, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        batch = input_shape[0]
        n_prior_box = self.n_base_anchor * input_shape[1] * input_shape[2]
        n_elements = 8

        return batch, n_prior_box, n_elements

    def call(self, inputs, **kwargs):
        f_height, f_width = inputs.shape[1:3]

        anchors = np.zeros((f_height, f_width, self.n_base_anchor, 4), np.float32)

        w_step = self.img_width / f_width
        h_step = self.img_height / f_height
        cx = np.linspace(0.5 * w_step, (0.5 + f_width - 1) * w_step, f_width)
        cy = np.linspace(0.5 * h_step, (0.5 + f_height - 1) * h_step, f_height)
        cx, cy = np.meshgrid(cx, cy)
        cx = np.tile(np.expand_dims(cx, axis=-1), (1, 1, self.n_base_anchor))
        cy = np.tile(np.expand_dims(cy, axis=-1), (1, 1, self.n_base_anchor))

        anchors[:, :, :, 0] = cx
        anchors[:, :, :, 1] = cy
        anchors[:, :, :, 2] = self.base_anchor[:, 0]
        anchors[:, :, :, 3] = self.base_anchor[:, 1]

        anchors = np.reshape(anchors, (-1, 4))
        anchors = change_wh(change_xy(anchors) / 300)
        anchors_tensor = K.expand_dims(anchors, 0)
        return K.tile(anchors_tensor, (K.shape(inputs)[0], 1, 1))

    def get_config(self):
        config = super(PriorBox, self).get_config()
        config.update({
            'img_height': self.img_height,
            'img_width': self.img_width,
            'base_anchors': self.base_anchor
        })

        return config