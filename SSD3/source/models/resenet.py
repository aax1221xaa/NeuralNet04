from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, Reshape, Concatenate, Softmax, Lambda, Activation, Add
from tensorflow.keras.layers import AvgPool2D
from tensorflow.keras.layers import BatchNormalization as BN
from tensorflow.keras.models import Model




class ResNet:
    @staticmethod
    def _ConvBlock(inCh, outCh, blockName):
        name = blockName + '_conv'

        def wrapper(x_input):
            x = Conv2D(inCh, (1, 1), padding='valid', name=name + '_1x1_1')(x_input)
            x = BN(name=name + '_bn_1')(x)
            x = Activation('relu', name=name + '_relu_1')(x)

            x = Conv2D(inCh, (3, 3), padding='same', name=name + '_3x3_1')(x)
            x = BN(name=name + '_bn_2')(x)
            x = Activation('relu', name=name + '_relu_2')(x)

            x = Conv2D(outCh, (1, 1), padding='valid', name=name + '_1x1_2')(x)
            x = BN(name=name + '_bn_3')(x)

            x2 = Conv2D(outCh, (1, 1), padding='valid', name=name + '_1x1_3')(x_input)
            x2 = BN(name=name + '_bn_4')(x2)

            x = Add(name=name + '_add')([x, x2])
            x = Activation('relu', name=name + '_relu_3')(x)

            return x
        return wrapper

    @staticmethod
    def _IdentyBlock(ch, blockName, nIdent):
        name = blockName + f'_ident{nIdent}'

        def wrapper(x_input):
            x = Conv2D(ch, (1, 1), padding='valid', name=name + '_1x1_1')(x_input)
            x = BN(name=name + '_bn_1')(x)
            x = Activation('relu', name=name + '_relu_1')(x)

            x = Conv2D(ch, (3, 3), padding='same', name=name + '_3x3_1')(x)
            x = BN(name=name + '_bn_2')(x)
            x = Activation('relu', name=name + '_relu_2')(x)

            x = Conv2D(ch, (1, 1), padding='valid', name=name + '_1x1_2')(x)
            x = BN(name=name + '_bn_3')(x)

            x = Add(name=name + '_add')([x, x_input])
            x = Activation('relu', name=name + '_relu_3')(x)

            return x
        return wrapper

    def _StageBlock(self, inCh, outCh, nStage, nIdentity):
        blockName = f'stage{nStage}'

        def wrapper(x_input):
            x = self._ConvBlock(inCh, outCh, blockName)(x_input)
            for n in range(nIdentity):
                x = self._IdentyBlock(outCh, blockName, n)(x)

            return x
        return wrapper

    def GetModel(self, x_size, addFC=False, **kwargs):
        channels = kwargs['channels']
        repeats = kwargs['repeat']

        x_input = Input(x_size, name='input')

        x = Conv2D(channels[0], (7, 7), (2, 2), padding='same', name='stage1_conv_7x7')(x_input)
        x = BN(name='stage1_bn')(x)
        x = Activation('relu', name='stage1_relu')(x)
        x = MaxPooling2D((3, 3), (2, 2), padding='valid', name='stage1_maxpool')(x)

        for stage, channel in enumerate(channels[1:]):
            x = self._StageBlock(channel[0], channel[1], stage, repeats[stage])(x)

        if addFC:
            x = AvgPool2D()

        return x