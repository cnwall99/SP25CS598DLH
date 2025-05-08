import tensorflow as tf
from keras.layers import Layer
from keras import initializers, regularizers, constraints

from keras.saving import register_keras_serializable

@register_keras_serializable(package="Custom")
class InstanceNormalization(Layer):
    def __init__(self,
                 axis=-1,
                 epsilon=1e-6,
                 center=True,
                 scale=True,
                 beta_initializer="zeros",
                 gamma_initializer="ones",
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis {} of input tensor should be defined. Found None.'.format(self.axis))

        if self.scale:
            self.gamma = self.add_weight(shape=(dim,),
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint,
                                         name='gamma')
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=(dim,),
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint,
                                        name='beta')
        else:
            self.beta = None
        super(InstanceNormalization, self).build(input_shape)

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=[1], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (inputs - mean) * inv

        if self.scale:
            normalized = normalized * self.gamma
        if self.center:
            normalized = normalized + self.beta
        return normalized

    def compute_output_shape(self, input_shape):
        return input_shape
