import keras
import tensorflow as tf
from .instance_norm import InstanceNormalization
from keras import Model
from keras.activations import sigmoid, relu, gelu
from keras.layers import Dense, Dropout, Reshape, LayerNormalization, MultiHeadAttention, Add, Flatten, Input, Layer, \
    GlobalAveragePooling1D, AveragePooling1D, Concatenate, SeparableConvolution1D, Conv1D
from keras.regularizers import L2

import tensorflow as tf
from keras import layers

import tensorflow as tf
from keras import layers

class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, inputs):
        # Get the shape of the input tensor
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.shape(inputs)[1]
        channels = tf.shape(inputs)[2]

        # Expand dimensions to create "height" for patches
        inputs_expanded = tf.expand_dims(inputs, axis=2)

        patches = tf.image.extract_patches(
            images=inputs_expanded,
            sizes=[1, self.patch_size, 1, 1],
            strides=[1, self.patch_size, 1, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )

        patch_dims = patches.shape[-1]
        return tf.reshape(patches, [batch_size, -1, patch_dims])


class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim, l2_weight, **kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.projection = Dense(units=projection_dim, kernel_regularizer=L2(l2_weight),
                                bias_regularizer=L2(l2_weight))
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=tf.shape(patch)[1], delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def mlp(x, hidden_units, dropout_rate, l2_weight):
    for units in hidden_units:
        x = Dense(units, activation=None, kernel_regularizer=L2(l2_weight), bias_regularizer=L2(l2_weight))(x)
        x = gelu(x)
        x = Dropout(dropout_rate)(x)
    return x


def create_transformer_model(input_shape, num_patches,
                              projection_dim, transformer_layers,
                              num_heads, transformer_units, mlp_head_units,
                              num_classes, drop_out, reg, l2_weight, demographic=False):
    if reg:
        activation = None
    else:
        activation = 'sigmoid'

    inputs = Input(shape=input_shape)
    patch_size = input_shape[0] // num_patches

    if demographic:
        normalized_inputs = InstanceNormalization(axis=-1, epsilon=1e-6, center=False, scale=False,
                                                   beta_initializer="glorot_uniform",
                                                   gamma_initializer="glorot_uniform")(inputs[:, :, :-1])
        demo = inputs[:, :12, -1]
    else:
        normalized_inputs = InstanceNormalization(axis=-1, epsilon=1e-6, center=False, scale=False,
                                                   beta_initializer="glorot_uniform",
                                                   gamma_initializer="glorot_uniform")(inputs)

    patches = Patches(patch_size=patch_size)(normalized_inputs)
    encoded_patches = PatchEncoder(num_patches=num_patches, projection_dim=projection_dim, l2_weight=l2_weight)(patches)

    for _ in range(transformer_layers):
        x1 = encoded_patches
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=drop_out,
            kernel_regularizer=L2(l2_weight), bias_regularizer=L2(l2_weight)
        )(x1, x1)
        x2 = Add()([attention_output, encoded_patches])
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, transformer_units, drop_out, l2_weight)
        encoded_patches = Add()([x3, x2])

    x = LayerNormalization(epsilon=1e-6)(encoded_patches)
    x = GlobalAveragePooling1D()(x)
    features = mlp(x, mlp_head_units, 0.0, l2_weight)

    logits = Dense(num_classes, kernel_regularizer=L2(l2_weight), bias_regularizer=L2(l2_weight),
                   activation=activation)(features)

    return Model(inputs=inputs, outputs=logits)


def create_hybrid_transformer_model(input_shape):
    transformer_units = [32, 32]
    transformer_layers = 2
    num_heads = 4
    l2_weight = 0.001
    drop_out = 0.25
    mlp_head_units = [256, 128]
    num_patches = 30
    projection_dim = 32

    # Calculate patch_size and ensure it's a valid size
    patch_size = input_shape[0] // num_patches
    patch_size = max(patch_size, 1)  # Ensure patch_size is at least 1

    if patch_size <= 0 or patch_size > input_shape[0]:
        raise ValueError(f"Invalid patch_size {patch_size} for input sequence length {input_shape[0]}.")

    input1 = Input(shape=input_shape)

    conv11 = Conv1D(16, 256)(input1)
    conv12 = Conv1D(16, 256)(input1)
    conv13 = Conv1D(16, 256)(input1)

    pwconv1 = SeparableConvolution1D(32, 1)(input1)
    pwconv2 = SeparableConvolution1D(32, 1)(pwconv1)

    conv21 = Conv1D(16, 256)(conv11)
    conv22 = Conv1D(16, 256)(conv12)
    conv23 = Conv1D(16, 256)(conv13)

    concat = keras.layers.concatenate([conv21, conv22, conv23], axis=-1)
    concat = Dense(64, activation=relu)(concat)
    concat = Dense(64, activation=sigmoid)(concat)
    concat = SeparableConvolution1D(32, 1)(concat)
    concat = keras.layers.concatenate([concat, pwconv2], axis=1)

    normalized_inputs = InstanceNormalization(axis=-1, epsilon=1e-6, center=False, scale=False,
                                               beta_initializer="glorot_uniform",
                                               gamma_initializer="glorot_uniform")(concat)

    # Now you can safely pass a valid patch_size to Patches
    patches = Patches(patch_size=patch_size)(normalized_inputs)
    encoded_patches = PatchEncoder(num_patches=num_patches, projection_dim=projection_dim, l2_weight=l2_weight)(patches)

    for _ in range(transformer_layers):
        x1 = encoded_patches
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=drop_out,
            kernel_regularizer=L2(l2_weight), bias_regularizer=L2(l2_weight)
        )(x1, x1)
        x2 = Add()([attention_output, encoded_patches])
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, transformer_units, drop_out, l2_weight)
        encoded_patches = Add()([x3, x2])

    x = LayerNormalization(epsilon=1e-6)(encoded_patches)
    x = GlobalAveragePooling1D()(x)
    features = mlp(x, mlp_head_units, 0.0, l2_weight)

    logits = Dense(1, kernel_regularizer=L2(l2_weight), bias_regularizer=L2(l2_weight),
                   activation='sigmoid')(features)

    model = Model(inputs=input1, outputs=logits)
    return model
