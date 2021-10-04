import numpy as np
import cv2

from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import Input
from source.config import config
from source.mic import change_wh, IOU, Anchor, change_xy, nms
from source.voc import voc_record
from source.models.backBone import PreInput
from source.models.mobileNet import MobileNet_v1, SSD300, GetLayers
from source.Geometric import RandomExpend, RandomMirror, RandomCrop, Expend
from source.Photometric import RandomBrightness, RandomContrast, ConvertColor, RandomSaturation, RandomHue, RandomChannelSwap
from source.PreProcessImage import PreProcessor


if __name__ == '__main__':
    C = config()

    anchorObj = Anchor(C)
    mobileNet = MobileNet_v1(Input([512, 512, 3], name='input'), (3, 3), (1, 1), 'conv', C)
    # mobileNet.LoadWeights('weights/PreTrain/mobileNetV1.h5')

    ssd300 = SSD300(
        anchorObj,
        GetLayers(
            mobileNet.GetModel(),
            [
                'input',
                'conv_pw_11_relu',
                'conv_pw_13_relu'
            ]),
        C
    )
    model_ssd300 = ssd300.GetModel()
























