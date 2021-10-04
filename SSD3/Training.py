from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, EarlyStopping
from tensorflow.keras.layers import Input

from source.config import *
from source.losses import *
from source.models.mobileNet import MobileNet_v1, SSD300, GetShape, GetLayers
from source.voc import *
from source.Geometric import *
from source.Photometric import RandomBrightness, RandomContrast, ConvertColor, RandomSaturation, RandomHue, RandomChannelSwap
from source.PreProcessImage import PreProcessor
from source.generator import Generator


if __name__ == '__main__':
    batch = 8
    epoch = 1
    epochLength = 1000

    C = config()
    anchorObj = Anchor(C)
    mobileNet = MobileNet_v1(Input([512, 512, 3], name='input'), (3, 3), (1, 1), 'conv', C)

    mobileNet.LoadWeights('weights/PreTrain/mobileNetV1.h5')

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
    #model_ssd300.load_weights('weights/ssd300_weights_1.hdf5')

    loss = MultiBoxLoss(C)

    sgd = SGD(learning_rate=0.001, momentum=0.9, decay=0.0, nesterov=False)
    model_ssd300.compile(optimizer=sgd, loss=loss.calc_loss)

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
    print(f_size)

    anchors = anchorObj.GetAnchors(f_size)

    train = voc_record('e:/data_set/voc2012train2.record', C)

    encoder = y_encoder(anchors, C)
    train_process = PreProcessor(
        [
            RandomContrast(),
            ConvertColor(cv2.COLOR_BGR2HSV),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(cv2.COLOR_HSV2BGR)
        ]
        ,
        [
            RandomMirror(),
            Expend(C)
            #RandomCrop(0.5, 50, 0.5, min_aspect_ratio=2 / 3, max_aspect_ratio=3 / 2)
        ],
        C
    )
    train_iter = train()
    train_generator = Generator(batch, train_iter, train_process, encoder)

    '''
    valid = voc_record('e:/data_set/voc2012val2.record', C)
    valid_process = PreProcessor(
        [
            RandomContrast(),
            ConvertColor(cv2.COLOR_BGR2HSV),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(cv2.COLOR_HSV2BGR)
        ]
        ,
        [
            RandomMirror(),
            Expend(C)
            # RandomCrop(0.5, 50, 0.5, min_aspect_ratio=2 / 3, max_aspect_ratio=3 / 2)
        ],
        C
    )
    valid_iter = valid()
    valid_generator = Generator(batch, valid_iter, valid_process, encoder)
    '''

    callback = [
        ModelCheckpoint(
            filepath='weights/ssd300_weights_1.hdf5',
            monitor='loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode='auto',
            period=1
        ),
        TerminateOnNaN(),
    ]

    model_ssd300.fit_generator(
        generator=train_generator(),
        steps_per_epoch=epochLength,
        epochs=epoch,
        callbacks=callback
    )





























