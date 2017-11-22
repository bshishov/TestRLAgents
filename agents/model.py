import numpy as np


def as_batch(x):
    return np.expand_dims(x, axis=0)


class Model(object):
    def fit(self, key, value):
        raise NotImplementedError

    def fit_batch(self, keys, values):
        raise NotImplementedError

    def predict(self, key):
        raise NotImplementedError

    def predict_batch(self, keys):
        raise NotImplementedError


class KerasModel(Model):
    def __init__(self, model):
        self.model = model

    def fit(self, key, value):
        self.model.train_on_batch(as_batch(key), as_batch(value))

    def fit_batch(self, keys, values):
        self.model.train_on_batch(keys, values)

    def predict(self, key):
        return self.model.predict(as_batch(key))[0]

    def predict_batch(self, keys):
        return self.model.predict(keys)


class SkLearnModel(Model):
    def __init__(self, model):
        self.model = model

    def fit(self, key, value):
        self.model.partial_fit(as_batch(key), as_batch(value))

    def fit_batch(self, keys, values):
        self.model.partial_fit(keys, values)

    def predict(self, key):
        return self.model.predict(as_batch(key))[0]

    def predict_batch(self, keys):
        return self.model.predict(keys)


class LinearRegression(Model):
    def __init__(self, input_shape, output_shape):
        pass

    def fit(self, key, value):
        raise NotImplementedError

    def fit_batch(self, keys, values):
        raise NotImplementedError

    def predict(self, key):
        raise NotImplementedError

    def predict_batch(self, keys):
        raise NotImplementedError