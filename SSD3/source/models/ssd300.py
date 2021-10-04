from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Reshape, Concatenate, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from source.layers import L2Norm, PriorBox

from source.models.backBone import BackBoneNet


class VGG16(BackBoneNet):
    def __init__(self, inputSize, anchorObj, C):
        super(VGG16, self).__init__(inputSize, C)

        self.base_anchors = anchorObj.baseAnchors
        self.n_anchors = [len(anchor) for anchor in anchorObj.baseAnchors]
        self.n_classes = len(C.classes) + 1

        self.fLayers = []

        conv1_1 = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv1_1')(self.x_out)
        conv1_2 = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv1_2')(conv1_1)
        pool1 = MaxPooling2D((2, 2), (2, 2), padding='same', name='pool1')(conv1_2)

        conv2_1 = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv2_1')(pool1)
        conv2_2 = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv2_2')(conv2_1)
        pool2 = MaxPooling2D((2, 2), (2, 2), padding='same', name='pool2')(conv2_2)

        conv3_1 = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv3_1')(pool2)
        conv3_2 = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv3_2')(conv3_1)
        conv3_3 = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv3_3')(conv3_2)
        pool3 = MaxPooling2D((2, 2), (2, 2), padding='same', name='pool3')(conv3_3)

        conv4_1 = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv4_1')(pool3)
        conv4_2 = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv4_2')(conv4_1)
        conv4_3 = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv4_3')(conv4_2)
        pool4 = MaxPooling2D((2, 2), (2, 2), padding='same', name='pool4')(conv4_3)

        conv5_1 = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv5_1')(pool4)
        conv5_2 = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv5_2')(conv5_1)
        conv5_3 = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv5_3')(conv5_2)
        pool5 = MaxPooling2D((3, 3), (1, 1), padding='same', name='pool5')(conv5_3)

        fc6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(self.C.l2_reg), activation='relu', name='fc6')(pool5)

        fc7 = Conv2D(1024, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(self.C.l2_reg), name='fc7')(fc6)

        conv6_1 = Conv2D(256, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(self.C.l2_reg), name='conv6_1')(fc7)
        conv6_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv6_padding')(conv6_1)
        conv6_2 = Conv2D(512, (3, 3), strides=(2, 2), padding='valid', activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(self.C.l2_reg), name='conv6_2')(conv6_1)

        conv7_1 = Conv2D(128, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(self.C.l2_reg), name='conv7_1')(conv6_2)
        conv7_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv7_padding')(conv7_1)
        conv7_2 = Conv2D(256, (3, 3), strides=(2, 2), padding='valid', activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(self.C.l2_reg), name='conv7_2')(conv7_1)

        conv8_1 = Conv2D(128, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(self.C.l2_reg), name='conv8_1')(conv7_2)
        conv8_2 = Conv2D(256, (3, 3), strides=(1, 1), padding='valid', activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(self.C.l2_reg), name='conv8_2')(conv8_1)

        conv9_1 = Conv2D(128, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(self.C.l2_reg), name='conv9_1')(conv8_2)
        conv9_2 = Conv2D(256, (3, 3), strides=(1, 1), padding='valid', activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(self.C.l2_reg), name='conv9_2')(conv9_1)

        conv4_3_norm = L2Norm(10, name='conv4_3_norm')(conv4_3)

        conv4_3_norm_mbox_conf = Conv2D(self.n_anchors[0] * self.n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(self.C.l2_reg),
                                        name='conv4_3_norm_mbox_conf')(conv4_3_norm)
        fc7_mbox_conf = Conv2D(self.n_anchors[1] * self.n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(self.C.l2_reg), name='fc7_mbox_conf')(fc7)
        conv6_2_mbox_conf = Conv2D(self.n_anchors[2] * self.n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(self.C.l2_reg), name='conv6_2_mbox_conf')(conv6_2)
        conv7_2_mbox_conf = Conv2D(self.n_anchors[3] * self.n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(self.C.l2_reg), name='conv7_2_mbox_conf')(conv7_2)
        conv8_2_mbox_conf = Conv2D(self.n_anchors[4] * self.n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(self.C.l2_reg), name='conv8_2_mbox_conf')(conv8_2)
        conv9_2_mbox_conf = Conv2D(self.n_anchors[5] * self.n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(self.C.l2_reg), name='conv9_2_mbox_conf')(conv9_2)

        conv4_3_norm_mbox_loc = Conv2D(self.n_anchors[0] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(self.C.l2_reg), name='conv4_3_norm_mbox_loc')(conv4_3_norm)
        fc7_mbox_loc = Conv2D(self.n_anchors[1] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(self.C.l2_reg), name='fc7_mbox_loc')(fc7)
        conv6_2_mbox_loc = Conv2D(self.n_anchors[2] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(self.C.l2_reg), name='conv6_2_mbox_loc')(conv6_2)
        conv7_2_mbox_loc = Conv2D(self.n_anchors[3] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(self.C.l2_reg), name='conv7_2_mbox_loc')(conv7_2)
        conv8_2_mbox_loc = Conv2D(self.n_anchors[4] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(self.C.l2_reg), name='conv8_2_mbox_loc')(conv8_2)
        conv9_2_mbox_loc = Conv2D(self.n_anchors[5] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(self.C.l2_reg), name='conv9_2_mbox_loc')(conv9_2)

        conv4_3_norm_mbox_priorbox = PriorBox(self.C.imageSize, self.base_anchors[0], name='conv4_3_norm_mbox_priorbox')(conv4_3_norm)
        fc7_mbox_priorbox = PriorBox(self.C.imageSize, self.base_anchors[1], name='fc7_mbox_priorbox')(fc7)
        conv6_2_mbox_priorbox = PriorBox(self.C.imageSize, self.base_anchors[2], name='conv6_2_mbox_priorbox')(conv6_2)
        conv7_2_mbox_priorbox = PriorBox(self.C.imageSize, self.base_anchors[3], name='conv7_2_mbox_priorbox')(conv7_2)
        conv8_2_mbox_priorbox = PriorBox(self.C.imageSize, self.base_anchors[4], name='conv8_2_mbox_priorbox')(conv8_2)
        conv9_2_mbox_priorbox = PriorBox(self.C.imageSize, self.base_anchors[5], name='conv9_2_mbox_priorbox')(conv9_2)

        conv4_3_norm_mbox_conf_reshape = Reshape((-1, self.n_classes), name='conv4_3_norm_mbox_conf_reshape')(conv4_3_norm_mbox_conf)
        fc7_mbox_conf_reshape = Reshape((-1, self.n_classes), name='fc7_mbox_conf_reshape')(fc7_mbox_conf)
        conv6_2_mbox_conf_reshape = Reshape((-1, self.n_classes), name='conv6_2_mbox_conf_reshape')(conv6_2_mbox_conf)
        conv7_2_mbox_conf_reshape = Reshape((-1, self.n_classes), name='conv7_2_mbox_conf_reshape')(conv7_2_mbox_conf)
        conv8_2_mbox_conf_reshape = Reshape((-1, self.n_classes), name='conv8_2_mbox_conf_reshape')(conv8_2_mbox_conf)
        conv9_2_mbox_conf_reshape = Reshape((-1, self.n_classes), name='conv9_2_mbox_conf_reshape')(conv9_2_mbox_conf)

        conv4_3_norm_mbox_loc_reshape = Reshape((-1, 4), name='conv4_3_norm_mbox_loc_reshape')(conv4_3_norm_mbox_loc)
        fc7_mbox_loc_reshape = Reshape((-1, 4), name='fc7_mbox_loc_reshape')(fc7_mbox_loc)
        conv6_2_mbox_loc_reshape = Reshape((-1, 4), name='conv6_2_mbox_loc_reshape')(conv6_2_mbox_loc)
        conv7_2_mbox_loc_reshape = Reshape((-1, 4), name='conv7_2_mbox_loc_reshape')(conv7_2_mbox_loc)
        conv8_2_mbox_loc_reshape = Reshape((-1, 4), name='conv8_2_mbox_loc_reshape')(conv8_2_mbox_loc)
        conv9_2_mbox_loc_reshape = Reshape((-1, 4), name='conv9_2_mbox_loc_reshape')(conv9_2_mbox_loc)

        conv4_3_norm_mbox_priorbox_reshape = Reshape((-1, 4), name='conv4_3_norm_mbox_priorbox_reshape')(conv4_3_norm_mbox_priorbox)
        fc7_mbox_priorbox_reshape = Reshape((-1, 4), name='fc7_mbox_priorbox_reshape')(fc7_mbox_priorbox)
        conv6_2_mbox_priorbox_reshape = Reshape((-1, 4), name='conv6_2_mbox_priorbox_reshape')(conv6_2_mbox_priorbox)
        conv7_2_mbox_priorbox_reshape = Reshape((-1, 4), name='conv7_2_mbox_priorbox_reshape')(conv7_2_mbox_priorbox)
        conv8_2_mbox_priorbox_reshape = Reshape((-1, 4), name='conv8_2_mbox_priorbox_reshape')(conv8_2_mbox_priorbox)
        conv9_2_mbox_priorbox_reshape = Reshape((-1, 4), name='conv9_2_mbox_priorbox_reshape')(conv9_2_mbox_priorbox)

        mbox_conf = Concatenate(axis=1, name='mbox_conf')([conv4_3_norm_mbox_conf_reshape,
                                                           fc7_mbox_conf_reshape,
                                                           conv6_2_mbox_conf_reshape,
                                                           conv7_2_mbox_conf_reshape,
                                                           conv8_2_mbox_conf_reshape,
                                                           conv9_2_mbox_conf_reshape])

        mbox_loc = Concatenate(axis=1, name='mbox_loc')([conv4_3_norm_mbox_loc_reshape,
                                                         fc7_mbox_loc_reshape,
                                                         conv6_2_mbox_loc_reshape,
                                                         conv7_2_mbox_loc_reshape,
                                                         conv8_2_mbox_loc_reshape,
                                                         conv9_2_mbox_loc_reshape])

        mbox_priorbox = Concatenate(axis=1, name='mbox_priorbox')([conv4_3_norm_mbox_priorbox_reshape,
                                                                   fc7_mbox_priorbox_reshape,
                                                                   conv6_2_mbox_priorbox_reshape,
                                                                   conv7_2_mbox_priorbox_reshape,
                                                                   conv8_2_mbox_priorbox_reshape,
                                                                   conv9_2_mbox_priorbox_reshape])

        mbox_conf_softmax = Softmax(name='mbox_conf_softmax')(mbox_conf)

        '''
        [batch, boxes, 1 + classes + 4 + 4]
        '''
        predict = Concatenate(axis=2, name='predictions')([mbox_conf_softmax, mbox_loc, mbox_priorbox])

        self.model_ssd = Model(self.x_input, predict)
        #return model_ssd, [conv4_3_norm_mbox_loc.shape, fc7_mbox_loc.shape, conv6_2_mbox_loc.shape, conv7_2_mbox_loc.shape, conv8_2_mbox_loc.shape, conv9_2_mbox_loc.shape]

    def GetModel(self):
        return self.model_ssd

    def GetShape(self, layerNames):
        shapes = []
        i = 0

        for layer in self.model_ssd.layers:
            if layer.name == layerNames[i]:
                shapes.append(layer.output_shape)
                i += 1

        if i != len(layerNames):
            return None
        else:
            return shapes































