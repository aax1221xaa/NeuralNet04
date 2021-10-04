import numpy as np
import cv2
from tensorflow.keras.layers import Input

from source.config import config
from source.mic import change_wh, IOU, Anchor, change_xy, nms
from source.voc import voc_record
from source.models.mobileNet import MobileNet_v1, SSD300, GetShape, GetLayers
from source.Geometric import RandomExpend, RandomMirror, RandomCrop, Expend
from source.Photometric import RandomBrightness, RandomContrast, ConvertColor, RandomSaturation, RandomHue, RandomChannelSwap
from source.PreProcessImage import PreProcessor


DEBUG = 0

if DEBUG == 0:
    if __name__ == '__main__':
        C = config()

        anchorsObj = Anchor(C)
        mobileNet = MobileNet_v1(Input([512, 512, 3], name='input'), (3, 3), (1, 1), 'conv', C)
        ssd300 = SSD300(
            anchorsObj,
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
        model_ssd300.load_weights('weights/ssd300_weights_2.hdf5', True)

        f_size = GetShape(
            model_ssd300,
            [
                'conv_pw_11_relu',
                'conv_pw_13_relu',
                'conv14_pw_relu',
                'conv15_pw_relu',
                'conv16_pw_relu',
                'conv17_pw_relu'
            ]
        )
        anchors = anchorsObj.GetAnchors(f_size)

        valid = voc_record('e:/data_set/voc2012val2.record', C)
        valid_iter = valid()

        preProcessor = PreProcessor(
            [],
            [
                Expend(C)
            ],
            C
        )

        for sample in valid_iter:
            image, _ = preProcessor(sample)
            X = np.expand_dims(image, 0)

            Y = model_ssd300.predict_on_batch(X)

            Y = np.squeeze(Y, 0)
            maxArg = np.argmax(Y[..., :-4], -1)
            valid_idx = np.where(maxArg != len(C.classes))[0]

            maxArg = maxArg[valid_idx]
            confidence = Y[valid_idx, :-4]
            regressor = Y[valid_idx, -4:] * C.variances

            anchors_wh = anchors[0][valid_idx]
            bestConf = np.max(confidence, -1)

            bbox_wh = np.zeros_like(regressor)
            bbox_wh[:, 0] = anchors_wh[:, 0] + anchors_wh[:, 2] * regressor[:, 0]
            bbox_wh[:, 1] = anchors_wh[:, 1] + anchors_wh[:, 3] * regressor[:, 1]
            bbox_wh[:, 2] = anchors_wh[:, 2] * np.exp(regressor[:, 2])
            bbox_wh[:, 3] = anchors_wh[:, 3] * np.exp(regressor[:, 3])
            bbox_xy = change_xy(bbox_wh)
            bbox_xy = np.around(bbox_xy).astype(np.int32)

            keep = nms(bestConf, bbox_xy, 0.4, 100)

            for xy, idx in zip(bbox_xy[keep], maxArg[keep]):
                cv2.rectangle(image, (xy[0], xy[1]), (xy[2], xy[3]), (0, 255, 255))
                cv2.putText(image, C.classes[idx], (xy[0], xy[1] + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0))

            cv2.imshow('RESULT', image)
            if cv2.waitKey() == 27:
                break
        cv2.destroyAllWindows()


if DEBUG == 1:

    def PreProcess(image, C):
        height, width = image.shape[:2]
        max_side = np.maximum(height, width)
        min_side = np.minimum(height, width)
        canvas = np.zeros((max_side, max_side, 3), np.uint8)

        start = (max_side - min_side) // 2
        canvas += np.array(C.meanBGR).astype(np.uint8)
        if width > height:
            canvas[start:start + height, :, :] = image
        else:
            canvas[:, start:start + width, :] = image

        return cv2.resize(canvas, (C.imageSize, C.imageSize), interpolation=cv2.INTER_CUBIC)

    if __name__ == '__main__':
        C = config()

        anchorsObj = Anchor(C)

        mobileNet = MobileNet_v1(Input([512, 512, 3], name='input'), (3, 3), (1, 1), 'conv', C)
        ssd300 = SSD300(
            anchorsObj,
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
        model_ssd300.load_weights('weights/ssd300_weights_3.hdf5', True)

        f_size = GetShape(
            model_ssd300,
            [
                'conv_pw_11_relu',
                'conv_pw_13_relu',
                'conv14_pw_relu',
                'conv15_pw_relu',
                'conv16_pw_relu',
                'conv17_pw_relu'
            ]
        )
        anchors = anchorsObj.GetAnchors(f_size)

        valid = voc_record('e:/data_set/voc2012val2.record', C)
        valid_iter = valid()

        preProcessor = PreProcessor(
            [],
            [
                Expend(C)
            ],
            C
        )

        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            success, frame = cap.read()
            if success:
                image = np.copy(frame)

                image = PreProcess(image, C)
                X = np.expand_dims(image, 0)

                Y = model_ssd300.predict_on_batch(X)

                Y = np.squeeze(Y, 0)
                maxArg = np.argmax(Y[..., :-4], -1)
                valid_idx = np.where(maxArg != len(C.classes))[0]

                maxArg = maxArg[valid_idx]
                confidence = Y[valid_idx, :-4]
                regressor = Y[valid_idx, -4:] * C.variances

                anchors_wh = anchors[0][valid_idx]
                bestConf = np.max(confidence, -1)

                bbox_wh = np.zeros_like(regressor)
                bbox_wh[:, 0] = anchors_wh[:, 0] + anchors_wh[:, 2] * regressor[:, 0]
                bbox_wh[:, 1] = anchors_wh[:, 1] + anchors_wh[:, 3] * regressor[:, 1]
                bbox_wh[:, 2] = anchors_wh[:, 2] * np.exp(regressor[:, 2])
                bbox_wh[:, 3] = anchors_wh[:, 3] * np.exp(regressor[:, 3])
                bbox_xy = change_xy(bbox_wh * C.imageSize)
                bbox_xy = np.around(bbox_xy).astype(np.int32)

                keep = nms(bestConf, bbox_xy, 0.4, 100)

                for xy, idx in zip(bbox_xy[keep], maxArg[keep]):
                    cv2.rectangle(image, (xy[0], xy[1]), (xy[2], xy[3]), (0, 255, 255))
                    cv2.putText(image, C.classes[idx], (xy[0], xy[1] + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0))

                cv2.imshow('RESULT', image)
                if cv2.waitKey(1) == 27:
                    break
        cap.release()
        cv2.destroyAllWindows()





























