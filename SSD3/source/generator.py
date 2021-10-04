import numpy as np


class Generator:
    def __init__(self, batch, iterator, preProcessor, encoder):
        self.batch = batch
        self.iterator = iterator
        self.preProcessor = preProcessor
        self.encoder = encoder

    def __call__(self):
        while True:
            X = []
            Y = []
            for i in range(self.batch):
                sample = next(self.iterator)
                image, labels = self.preProcessor(sample)

                X.append(image)
                Y.append(self.encoder(labels))
            yield np.stack(X), np.stack(Y)