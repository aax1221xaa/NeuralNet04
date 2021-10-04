class config:
    def __init__(self):
        self.classes = [
            'person',
            'bird',
            'cat',
            'cow',
            'dog',
            'horse',
            'sheep',
            'aeroplane',
            'bicycle',
            'boat',
            'bus',
            'car',
            'motorbike',
            'train',
            'bottle',
            'chair',
            'diningtable',
            'pottedplant',
            'sofa',
            'tvmonitor'
        ]
        self.imageSize = 512
        self.meanBGR = [123, 117, 104]
        self.scale_range = [0.2, 1.05]
        self.n_scale = 7
        self.ratios = [
            [1.0, 2.0, 0.5],
            [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
            [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
            [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
            [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
            [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0]
        ]
        self.variances = [0.1, 0.1, 0.2, 0.2]
        self.max_thresh = 0.5

        self.alpha = 1.0
        self.neg_pos_ratio = 3

        self.l2_reg = 0.0005


