import cv2
import numpy as np


class RandomBrightness:
    def __init__(self, prob=0.5, beta= 32):
        self.prob = prob
        self.beta = beta

    def __call__(self, image, labels):
        if (1 - self.prob) <= np.random.uniform(0, 1):
            beta = np.random.uniform(-self.beta, self.beta)
            image = cv2.convertScaleAbs(image, alpha= 1.0, beta= beta)

        return image, labels


class RandomContrast:
    def __init__(self, lower=0.5, upper=1.5, prob=0.5):
        self.lower = lower
        self.upper = upper
        self.prob = prob

    def __call__(self, image, labels):
        if (1 - self.prob) <= np.random.uniform(0, 1):
            alpha = np.random.uniform(self.lower, self.upper)
            image = cv2.convertScaleAbs(image, alpha= alpha, beta= 0.0)

        return image, labels


class ConvertColor:
    def __init__(self, change_code):
        self.change_code = change_code

    def __call__(self, image, labels):
        image = cv2.cvtColor(image, self.change_code)

        return image, labels


class RandomSaturation:
    def __init__(self, lower=0.3, upper=2.0, prob=0.5):
        self.lower = lower
        self.upper = upper
        self.prob = prob

    def __call__(self, image, labels):
        if (1 - self.prob) <= np.random.uniform(0, 1):
            factor = np.random.uniform(self.lower, self.upper)
            image[:, :, 1] = np.clip(image[:, :, 1] * factor, 0, 255)

        return image, labels


class RandomHue:
    def __init__(self, beta = 18, prob=0.5):
        self.beta = beta
        self.prob = prob

    def __call__(self, image, labels):
        if (1 - self.prob) <= np.random.uniform(0, 1):
            beta = np.random.uniform(-self.beta, self.beta)
            image[:, :, 0] = (image[:, :, 0] + beta) % 180.0

        return image, labels


class RandomChannelSwap:
    def __init__(self, prob=0.5):
        self.permutations = (
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0)
        )
        self.prob = prob

    def __call__(self, image, labels):
        if (1 - self.prob) <= np.random.uniform(0, 1):
            idx = np.random.randint(5)
            image = image[:, :, self.permutations[idx]]

            return image, labels