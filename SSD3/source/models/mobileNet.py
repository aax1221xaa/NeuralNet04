from tensorflow.keras.layers import Input, Conv2D, Concatenate, Reshape, Softmax, ZeroPadding2D
from tensorflow.keras.layers import Reshape, Concatenate, Softmax, Lambda, Activation, Add
from tensorflow.keras.layers import DepthwiseConv2D as DWConv2D
from tensorflow.keras.layers import BatchNormalization as BN
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K


class MobileNet_v1:
    def __init__(self, x_input, dwK, pwK, name, C):
        self.dwK = dwK
        self.pwK = pwK
        self.name = name

        def mean_normalizeation(tensor):
            return tensor - C.meanBGR

        def swap_channels(tensor):
            return K.stack([tensor[..., 2], tensor[..., 1], tensor[..., 0]], axis=-1)

        x = Lambda(mean_normalizeation, name='mean_normalization')(x_input)
        x = Lambda(swap_channels, name='swap_channels')(x)

        x = self.ConvBN(32, 2, 'same', 1)(x)
        x = self.ConvBlock(64, 1, 'same', 1)(x)
        x = self.ConvBlock(128, 2, 'same', 2)(x)
        x = self.ConvBlock(128, 1, 'same', 3)(x)
        x = self.ConvBlock(256, 2, 'same', 4)(x)
        x = self.ConvBlock(256, 1, 'same', 5)(x)
        x = self.ConvBlock(512, 2, 'same', 6)(x)
        x = self.ConvBlock(512, 1, 'same', 7)(x)
        x = self.ConvBlock(512, 1, 'same', 8)(x)
        x = self.ConvBlock(512, 1, 'same', 9)(x)
        x = self.ConvBlock(512, 1, 'same', 10)(x)
        x = self.ConvBlock(512, 1, 'same', 11)(x)
        x = self.ConvBlock(1024, 2, 'same', 12)(x)
        y_output = self.ConvBlock(1024, 1, 'same', 13)(x)

        self.model = Model(x_input, y_output)

    def _BN_ReLU(self, layerName):
        def wrapper(x_input):
            x = BN(name=layerName + '_bn')(x_input)
            x = Activation('relu', name=layerName + '_relu')(x)

            return x
        return wrapper

    def GetModel(self):
        return self.model

    def LoadWeights(self, path):
        self.model.load_weights(path, True)

    def ConvBN(self, outChannels, stride, padding, nLayer):
        stride = (stride, stride)
        layerName = f'{self.name}{nLayer}'

        def wrapper(x_input):
            x = Conv2D(outChannels, self.dwK, stride, padding=padding, use_bias=False, name=layerName)(x_input)
            x = self._BN_ReLU(layerName)(x)
            return x
        return wrapper

    def ConvBlock(self, outChannels, stride, padding, nLayer):
        def wrapper(x_input):
            x = self._DWConv2D(stride, padding, nLayer)(x_input)
            x = self._PWConv2D(outChannels, nLayer)(x)
            return x
        return wrapper

    def _PWConv2D(self, outCh, count):
        def wrapper(x_input):
            name = '{name}_pw_{count}'.format(name=self.name, count=count)
            x = Conv2D(outCh, self.pwK, use_bias=False, name=name)(x_input)
            x = self._BN_ReLU(name)(x)

            return x
        return wrapper

    def _DWConv2D(self, stride, padding, count):
        stride = (stride, stride)

        def wrapper(x_input):
            name = '{name}_dw_{count}'.format(name=self.name, count=count)
            x = DWConv2D(self.dwK, stride, padding, use_bias=False, name=name)(x_input)
            x = self._BN_ReLU(name)(x)

            return x
        return wrapper


class SSD300:
    def __init__(self, anchorObj, layers, C):
        self.base_anchors = anchorObj.baseAnchors
        self.n_anchors = [len(anchor) for anchor in anchorObj.baseAnchors]
        self.n_classes = len(C.classes) + 1

        x_input = layers[0]     # [None, 512, 512, 3]
        branch1 = layers[1]     # [None, 32, 32, 512]
        branch2 = layers[2]     # [None, 16, 16, 1024]

        x = DWConv2D((3, 3), (2, 2), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(C.l2_reg), use_bias=False, name='conv14_dw')(branch2)
        x = BN(name='conv14_dw_bn')(x)
        x = Activation('relu', name='conv14_dw_relu')(x)
        x = Conv2D(512, (1, 1), kernel_initializer='he_normal', kernel_regularizer=l2(C.l2_reg), use_bias=False, name='conv14_pw')(x)
        x = BN(name='conv14_pw_bn')(x)
        branch3 = Activation('relu', name='conv14_pw_relu')(x)

        x = DWConv2D((3, 3), (2, 2), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(C.l2_reg), use_bias=False, name='conv15_dw')(branch3)
        x = BN(name='conv15_dw_bn')(x)
        x = Activation('relu', name='conv15_dw_relu')(x)
        x = Conv2D(256, (1, 1), kernel_initializer='he_normal', kernel_regularizer=l2(C.l2_reg), use_bias=False, name='conv15_pw')(x)
        x = BN(name='conv15_pw_bn')(x)
        branch4 = Activation('relu', name='conv15_pw_relu')(x)

        x = DWConv2D((3, 3), (2, 2), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(C.l2_reg), use_bias=False, name='conv16_dw')(branch4)
        x = BN(name='conv16_dw_bn')(x)
        x = Activation('relu', name='conv16_dw_relu')(x)
        x = Conv2D(256, (1, 1), kernel_initializer='he_normal', kernel_regularizer=l2(C.l2_reg), use_bias=False, name='conv16_pw')(x)
        x = BN(name='conv16_pw_bn')(x)
        branch5 = Activation('relu', name='conv16_pw_relu')(x)

        x = DWConv2D((3, 3), (2, 2), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(C.l2_reg), use_bias=False, name='conv17_dw')(branch5)
        x = BN(name='conv17_dw_bn')(x)
        x = Activation('relu', name='conv17_dw_relu')(x)
        x = Conv2D(128, (1, 1), kernel_initializer='he_normal', kernel_regularizer=l2(C.l2_reg), use_bias=False, name='conv17_pw')(x)
        x = BN(name='conv17_pw_bn')(x)
        branch6 = Activation('relu', name='conv17_pw_relu')(x)

        '''
        x = Conv2D(256, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(C.l2_reg), name='conv14_1')(branch2)
        #x = BN(name='conv14_1_bn')(x)
        x = Activation('relu', name='conv14_1_relu')(x)

        x = Conv2D(512, (3, 3), (2, 2), padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(C.l2_reg), name='conv14_2')(x)
        #x = BN(name='conv14_2_bn')(x)
        branch3 = x = Activation('relu', name='conv14_2_relu')(x)       # [None, 8, 8, 512]

        x = Conv2D(128, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(C.l2_reg), name='conv15_1')(branch3)
        #x = BN(name='conv15_1_bn')(x)
        x = Activation('relu', name='conv15_1_relu')(x)

        x = Conv2D(256, (3, 3), (2, 2), padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(C.l2_reg), name='conv15_2')(x)
        #x = BN(name='conv15_2_bn')(x)
        branch4 = Activation('relu', name='conv15_2_relu')(x)           # [None, 4, 4, 256]

        x = Conv2D(128, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(C.l2_reg), name='conv16_1')(branch4)
        #x = BN(name='conv16_1_bn')(x)
        x = Activation('relu', name='conv16_1_relu')(x)

        x = Conv2D(256, (3, 3), (2, 2), padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(C.l2_reg), name='conv16_2')(x)
        #x = BN(name='conv16_2_bn')(x)
        branch5 = Activation('relu', name='conv16_2_relu')(x)           # [None, 2, 2, 256]

        #x = ZeroPadding2D((1, 1), name='conv16_2_zeropad')(branch5)
        x = Conv2D(64, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(C.l2_reg), name='conv17_1')(branch5)
        #x = BN(name='conv17_1_bn')(x)
        x = Activation('relu', name='conv17_1_relu')(x)

        x = Conv2D(128, (3, 3), (2, 2), padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(C.l2_reg), name='conv17_2')(x)
        #x = BN(name='conv17_2_bn')(x)
        branch6 = Activation('relu', name='conv17_2_relu')(x)           # [None, 1, 1, 128]
        '''
        conv11_conf = Conv2D(self.n_anchors[0] * self.n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(C.l2_reg), name='conv11_conf')(branch1)
        conv13_conf = Conv2D(self.n_anchors[1] * self.n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(C.l2_reg), name='conv13_conf')(branch2)
        conv14_2_conf = Conv2D(self.n_anchors[2] * self.n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(C.l2_reg), name='conv14_2_conf')(branch3)
        conv15_2_conf = Conv2D(self.n_anchors[3] * self.n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(C.l2_reg), name='conv15_2_conf')(branch4)
        conv16_2_conf = Conv2D(self.n_anchors[4] * self.n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(C.l2_reg), name='conv16_2_conf')(branch5)
        conv17_2_conf = Conv2D(self.n_anchors[5] * self.n_classes, (1, 1), padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(C.l2_reg), name='conv17_2_conf')(branch6)

        conv11_loc = Conv2D(self.n_anchors[0] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(C.l2_reg), name='conv11_loc')(branch1)
        conv13_loc = Conv2D(self.n_anchors[1] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(C.l2_reg), name='conv13_loc')(branch2)
        conv14_2_loc = Conv2D(self.n_anchors[2] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(C.l2_reg), name='conv14_2_loc')(branch3)
        conv15_2_loc = Conv2D(self.n_anchors[3] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(C.l2_reg), name='conv15_2_loc')(branch4)
        conv16_2_loc = Conv2D(self.n_anchors[4] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(C.l2_reg), name='conv16_2_loc')(branch5)
        conv17_2_loc = Conv2D(self.n_anchors[5] * 4, (1, 1), padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(C.l2_reg), name='conv17_2_loc')(branch6)

        conv11_conf_reshape = Reshape((-1, self.n_classes), name='conv11_conf_reshape')(conv11_conf)
        conv13_conf_reshape = Reshape((-1, self.n_classes), name='conv13_conf_reshape')(conv13_conf)
        conv14_2_conf_reshape = Reshape((-1, self.n_classes), name='conv14_2_conf_reshape')(conv14_2_conf)
        conv15_2_conf_reshape = Reshape((-1, self.n_classes), name='conv15_2_conf_reshape')(conv15_2_conf)
        conv16_2_conf_reshape = Reshape((-1, self.n_classes), name='conv16_2_conf_reshape')(conv16_2_conf)
        conv17_2_conf_reshape = Reshape((-1, self.n_classes), name='conv17_2_conf_reshape')(conv17_2_conf)

        conv11_loc_reshape = Reshape((-1, 4), name='conv11_loc_reshape')(conv11_loc)
        conv13_loc_reshape = Reshape((-1, 4), name='conv13_loc_reshape')(conv13_loc)
        conv14_2_loc_reshape = Reshape((-1, 4), name='conv14_2_loc_reshape')(conv14_2_loc)
        conv15_2_loc_reshape = Reshape((-1, 4), name='conv15_2_loc_reshape')(conv15_2_loc)
        conv16_2_loc_reshape = Reshape((-1, 4), name='conv16_2_loc_reshape')(conv16_2_loc)
        conv17_2_loc_reshape = Reshape((-1, 4), name='conv17_2_loc_reshape')(conv17_2_loc)

        '''
        conv11_priorbox_reshape = Reshape((-1, 4), name='conv11_priorbox_reshape')(conv11_priorbox)
        conv13_priorbox_reshape = Reshape((-1, 4), name='conv13_priorbox_reshape')(conv13_priorbox)
        conv14_2_priorbox_reshape = Reshape((-1, 4), name='conv14_2_priorbox_reshape')(conv14_2_priorbox)
        conv15_2_priorbox_reshape = Reshape((-1, 4), name='conv15_2_priorbox_reshape')(conv15_2_priorbox)
        conv16_2_priorbox_reshape = Reshape((-1, 4), name='conv16_2_priorbox_reshape')(conv16_2_priorbox)
        conv17_2_priorbox_reshape = Reshape((-1, 4), name='conv17_2_priorbox_reshape')(conv17_2_priorbox)
        '''

        confidence = Concatenate(1, name='confidence')([
            conv11_conf_reshape,
            conv13_conf_reshape,
            conv14_2_conf_reshape,
            conv15_2_conf_reshape,
            conv16_2_conf_reshape,
            conv17_2_conf_reshape
        ])

        location = Concatenate(1, name='location')([
            conv11_loc_reshape,
            conv13_loc_reshape,
            conv14_2_loc_reshape,
            conv15_2_loc_reshape,
            conv16_2_loc_reshape,
            conv17_2_loc_reshape
        ])

        '''
        priorbox = Concatenate(1, name='priorbox')([
            conv11_priorbox_reshape,
            conv13_priorbox_reshape,
            conv14_2_priorbox_reshape,
            conv15_2_priorbox_reshape,
            conv16_2_priorbox_reshape,
            conv17_2_priorbox_reshape
        ])
        '''

        conf_softmax = Softmax(name='conf_softmax')(confidence)
        predict = Concatenate(2, name='prediction')([conf_softmax, location])

        self.model_ssd = Model(x_input, predict)

        print('branc1 = ', branch1.shape)
        print('branc2 = ', branch2.shape)
        print('branc3 = ', branch3.shape)
        print('branc4 = ', branch4.shape)
        print('branc5 = ', branch5.shape)
        print('branc6 = ', branch6.shape)

    def GetModel(self):
        return self.model_ssd

def GetShape(model, layerNames):
    shape = []
    i = 0

    for layer in model.layers:
        if layer.name == layerNames[i]:
            shape.append(layer.output_shape)
            i += 1
        if i == len(layerNames):
            break

    if i != len(layerNames):
        return None
    else:
        return shape

def GetLayers(model, layerNames):
    layers = []
    i = 0

    for layer in model.layers:
        if layer.name == layerNames[i]:
            layers.append(layer.output)
            i += 1
        if i == len(layerNames):
            break

    if i != len(layerNames):
        return None
    else:
        return layers



























