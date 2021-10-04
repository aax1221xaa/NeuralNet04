from lxml import etree as et
import cv2
import numpy as np
import os
import tensorflow as tf


class voc_record():
    def __init__(self, path, C):
        self.path = path
        self.C = C


    @staticmethod
    def tf_record_extract(serialized_data):
        key_to_features = {
            'image/img_source': tf.io.FixedLenFeature((), tf.string, default_value= ''),
            'image/height': tf.io.FixedLenFeature((), tf.int64, 0),
            'image/width': tf.io.FixedLenFeature((), tf.int64, 0),
            'image/object/name': tf.io.FixedLenFeature((), tf.string, default_value=''),
            'image/object/x1':  tf.io.VarLenFeature(tf.int64),
            'image/object/y1':  tf.io.VarLenFeature(tf.int64),
            'image/object/x2':  tf.io.VarLenFeature(tf.int64),
            'image/object/y2':  tf.io.VarLenFeature(tf.int64)
        }

        features = tf.io.parse_single_example(serialized= serialized_data, features= key_to_features)

        img = tf.cast(features['image/img_source'], tf.string)
        height = tf.cast(features['image/height'], tf.int64)
        width = tf.cast(features['image/width'], tf.int64)

        classes = tf.cast(features['image/object/name'], tf.string)
        x1 = tf.cast(features['image/object/x1'], tf.int64)
        y1 = tf.cast(features['image/object/y1'], tf.int64)
        x2 = tf.cast(features['image/object/x2'], tf.int64)
        y2 = tf.cast(features['image/object/y2'], tf.int64)

        return {
            'image': img,
            'height': height,
            'width': width,
            'classes': classes,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2
        }

    def __call__(self, min_ratio=0.5):
        data_set = tf.data.TFRecordDataset(self.path, compression_type= 'GZIP').map(voc_record.tf_record_extract)
        iterator = data_set.batch(1)

        def wrapper():
            while True:
                for data in iterator:
                    img = data['image']
                    width = data['width'][0]
                    height = data['height'][0]
                    classes = data['classes'][0]
                    x1 = data['x1'].values - 1
                    y1 = data['y1'].values - 1
                    x2 = data['x2'].values - 1
                    y2 = data['y2'].values - 1

                    img_ratio = height / width
                    if img_ratio < min_ratio or img_ratio > (1 / min_ratio):
                        continue

                    img = cv2.imdecode(np.fromstring(img.numpy()[0], np.uint8), -1)
                    img = np.reshape(img, (height, width, 3))
                    bbox = np.column_stack([x1, y1, x2, y2])
                    classes = classes.numpy().decode().split()
                    label = np.expand_dims([self.C.classes.index(st) for st in classes], 1)

                    yield img, np.concatenate([bbox, label], axis=1)
                    # img, [n_bbox, 4 + 1]
        return wrapper()






































