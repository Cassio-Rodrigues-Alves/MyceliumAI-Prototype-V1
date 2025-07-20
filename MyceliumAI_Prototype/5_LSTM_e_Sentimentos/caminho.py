# caminho.py
import tensorflow as tf
from tensorflow.keras import layers

class Caminho(layers.Layer):
    def __init__(self, name, **kwargs):
        super(Caminho, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        # Multiplicação e adição elemento a elemento, os pesos são vetores.
        self.kernel = self.add_weight(
            shape=(input_shape[-1],),
            initializer="glorot_uniform",
            trainable=True,
            name="kernel"
        )
        self.bias = self.add_weight(
            shape=(input_shape[-1],),
            initializer="zeros",
            trainable=True,
            name="bias"
        )

    def call(self, inputs):
        return (inputs * self.kernel) + self.bias

    def get_config(self):
        config = super(Caminho, self).get_config()
        return config