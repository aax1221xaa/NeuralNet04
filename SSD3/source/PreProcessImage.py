import cv2
import numpy as np


class PreProcessor:
    def __init__(self, photometric, geometric, C):
        self.photometric = photometric
        self.geometric = geometric
        self.C = C

    def __call__(self, samples):
        image, labels = samples

        for photo in self.photometric:
            image, labels = photo(image, labels)

        image = image.astype(np.float32)
        for geo in self.geometric:
            image, labels = geo(image, labels)

        return self._Resize(image, labels)

    def _Resize(self, image, labels):
        image_h, image_w = image.shape[:2]

        w_ratio = self.C.imageSize / image_w
        h_ratio = self.C.imageSize / image_h

        bboxes = labels[:, :-1]
        bboxes[:, [0, 2]] = np.round(bboxes[:, [0, 2]].astype(np.float32) * w_ratio).astype(np.int32)
        bboxes[:, [1, 3]] = np.round(bboxes[:, [1, 3]].astype(np.float32) * h_ratio).astype(np.int32)
        image = cv2.resize(image, (self.C.imageSize, self.C.imageSize), cv2.INTER_CUBIC)
        labels[:, :-1] = bboxes

        return image, labels

'''
class ScaleImage:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __call__(self, indata):
        height, width = indata['image'].shape[:2]
        w_ratio = self.width / width
        h_ratio = self.height / height

        indata['label'][:, [1, 3]] = np.round((indata['label'][:, [1, 3]].astype(np.float32) * w_ratio)).astype(np.int32)
        indata['label'][:, [2, 4]] = np.round((indata['label'][:, [2, 4]].astype(np.float32) * h_ratio)).astype(np.int32)
        indata['image'] = cv2.resize(indata['image'], (self.width, self.height))
        return indata
'''
