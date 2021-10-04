from source.mic import *
import cv2


class RandomExpend:
    def __init__(self, C, prob=0.5, min_ratio=1, max_ratio=4):
        self.meanBGR = C.meanBGR
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.prob = prob

    def __call__(self, image, labels):
        img_height, img_width = image.shape[:2]

        if (1 - self.prob) <= np.random.uniform(0, 1):
            ratio = np.random.uniform(self.min_ratio, self.max_ratio)
            ex_width = np.round(img_width * ratio).astype(np.int32)
            ex_height = np.round(img_height * ratio).astype(np.int32)
            left = np.random.randint(ex_width - img_width)
            top = np.random.randint(ex_height - img_height)

            eximg = np.zeros((ex_height, ex_width, 3), np.float32) + self.meanBGR
            eximg[top:top + img_height, left:left + img_width, :] = image

            labels[:, :-1] += (left, top, left, top)

            return eximg, labels
        else:
            return image, labels

class Expend:
    def __init__(self, C):
        self.bg_color = C.meanBGR

    def __call__(self, image, labels):
        height, width = image.shape[:2]
        max_side = np.maximum(height, width)
        min_side = np.minimum(height, width)
        canvas = np.zeros((max_side, max_side, 3), np.uint8)

        start = (max_side - min_side) // 2
        canvas += np.array(self.bg_color).astype(np.uint8)
        if width > height:
            canvas[start:start + height, :, :] = image
            labels[:, [1, 3]] += start
        else:
            canvas[:, start:start + width, :] = image
            labels[:, [0, 2]] += start
        image = canvas

        return image, labels

class RandomCrop:
    def __init__(self,
                 prob=0.5,
                 n_trial=50,
                 min_scale=0.3,
                 max_scale=0.9,
                 min_aspect_ratio=0.5,
                 max_aspect_ratio=2.0):

        self.prob = prob
        self.n_trial = np.maximum(1, n_trial)
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio

    def _GetWindow(self, image_w, image_h):
        while True:
            w = np.random.uniform(self.min_scale, self.max_scale) * image_w
            h = np.random.uniform(self.min_scale, self.max_scale) * image_h

            w = np.round(w).astype(np.int32)
            h = np.round(h).astype(np.int32)

            if (w / h) < self.min_aspect_ratio or (w / h) > self.max_aspect_ratio:
                continue
            else:
                break

        x = np.random.randint(0, image_w - w)
        y = np.random.randint(0, image_h - h)

        return np.array([x, y, w, h])

    def __call__(self, image, labels):
        image_h, image_w = image.shape[:2]

        if (1 - self.prob) <= np.random.uniform(0, 1):
            for _ in range(self.n_trial):
                x, y, w, h = self._GetWindow(image_w, image_h)

                newLabels = labels[:, :-1] - [x, y, x, y]
                newLabels = np.maximum(0, newLabels)
                newLabels[:, 0] = np.minimum(w, newLabels[:, 0])
                newLabels[:, 1] = np.minimum(h, newLabels[:, 1])
                newLabels[:, 2] = np.minimum(w, newLabels[:, 2])
                newLabels[:, 3] = np.minimum(h, newLabels[:, 3])

                newArea = (newLabels[:, 2] - newLabels[:, 0]) * (newLabels[:, 3] - newLabels[:, 1])
                oldArea = (labels[:, 2] - labels[:, 0]) * (labels[:, 3] - labels[:, 1])

                validMark = newArea / oldArea > 0.7

                if np.count_nonzero(validMark) == 0:
                    continue

                labels[validMark, :-1] = newLabels[validMark]
                labels = labels[validMark]
                image = image[y:y+h, x:x+w]
                break

        return image, labels


class RandomMirror:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, labels):
        if (1 - self.prob) <= np.random.uniform(0, 1):
            mode = np.random.randint(0, 2)

            if mode == 0:
                image, labels = self.h_mirror(image, labels)
            elif mode == 1:
                image, labels = self.v_mirror(image, labels)
            elif mode == 2:
                image, labels = self.h_mirror(image, labels)
                image, labels = self.v_mirror(image, labels)

        return image, labels

    def h_mirror(self, image, labels):
        width = image.shape[1]

        image = cv2.flip(image, 1)
        labels[:, [2, 0]] = width - labels[:, [0, 2]]

        return image, labels

    def v_mirror(self, image, labels):
        height = image.shape[0]

        image[:] = cv2.flip(image, 0)
        labels[:, [3, 1]] = height - labels[:, [1, 3]]

        return image, labels
