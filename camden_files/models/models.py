import keras
from keras import Input, Model
from keras.layers import Dense, Flatten, MaxPooling2D, Conv2D, BatchNormalization, LSTM, Bidirectional, Permute, \
    Reshape, GRU, Conv1D, MaxPooling1D, Activation, Dropout, GlobalAveragePooling1D, multiply, MultiHeadAttention, Add, \
    LayerNormalization, SeparableConvolution1D
from keras.models import Sequential
from keras.activations import relu, sigmoid, gelu
from keras.regularizers import l2
from .instance_norm import InstanceNormalization
from .transformer import create_transformer_model, mlp, create_hybrid_transformer_model


def create_cnn_model(input_shape):
    model = Sequential()
    for i in range(3):  # Reduce number of pooling layers
        model.add(Conv1D(45, 3, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation(relu))
        model.add(MaxPooling1D(pool_size=2))  # pool_size=2, will reduce time dimension

    model.add(Flatten())
    for i in range(2):
        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(Activation(relu))
        model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    return model



def create_cnnlstm_model(input_a_shape, weight=1e-3):
    cnn_filters = 32  # 128
    cnn_kernel_size = 3  # Reduced kernel size from 4 to 3
    input1 = Input(shape=input_a_shape)
    input1 = InstanceNormalization(axis=-1, epsilon=1e-6, center=False, scale=False, 
                                   beta_initializer="glorot_uniform", gamma_initializer="glorot_uniform")(input1)
    
    # Reduced kernel size and ensured padding='same' to preserve dimensions
    x1 = Conv1D(cnn_filters, cnn_kernel_size, activation='relu', padding='same')(input1)
    x1 = Conv1D(cnn_filters, cnn_kernel_size, activation='relu', padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling1D(pool_size=1)(x1)

    # More layers with the same padding strategy
    x1 = Conv1D(cnn_filters, cnn_kernel_size, activation='relu', padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling1D()(x1)

    x1 = Conv1D(cnn_filters, cnn_kernel_size, activation='relu', padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling1D()(x1)

    # LSTM layers
    x1 = LSTM(32, return_sequences=True)(x1)  # 256
    x1 = LSTM(32, return_sequences=True)(x1)  # 256
    x1 = LSTM(32)(x1)  # 256
    x1 = Flatten()(x1)

    # Dense layers
    x1 = Dense(32, activation='relu')(x1)  # 64
    x1 = Dense(32, activation='relu')(x1)  # 64
    outputs = Dense(1, activation='sigmoid')(x1)

    model = Model(inputs=input1, outputs=outputs)
    return model



def create_semscnn_model(input_a_shape):
    input1 = Input(shape=input_a_shape)

    # Modified Conv1D layers for 11 timesteps
    x1 = Conv1D(45, 3, strides=1, padding='same')(input1)  # kernel_size=3, padding='same'
    x1 = Conv1D(45, 3, strides=2, padding='same')(x1)  # kernel_size=3, padding='same'
    x1 = BatchNormalization()(x1)
    x1 = Activation(relu)(x1)
    x1 = MaxPooling1D()(x1)

    x1 = Conv1D(45, 3, strides=2, padding='same')(x1)  # kernel_size=3, padding='same'
    x1 = BatchNormalization()(x1)
    x1 = Activation(relu)(x1)
    x1 = MaxPooling1D()(x1)

    x1 = Conv1D(45, 3, strides=2, padding='same')(x1)  # kernel_size=3, padding='same'
    x1 = BatchNormalization()(x1)
    x1 = Activation(relu)(x1)
    x1 = MaxPooling1D(pool_size=1)(x1)

    # Squeeze-and-excitation mechanism
    squeeze = Flatten()(x1)
    excitation = Dense(128, activation='relu')(squeeze)
    excitation = Dense(64, activation='relu')(excitation)
    logits = Dense(1, activation='sigmoid')(excitation)
    
    model = Model(inputs=input1, outputs=logits)
    return model






def get_model(config):
    input_shape = (config["input_length"], len(config["channels"]))
    inputs = Input(shape=input_shape)

    # Select appropriate model function from dictionary
    model_name  = config["model_name"]
    if "cnn" in model_name and "lstm" in model_name:
        model= create_cnnlstm_model(input_shape)
    elif "cnn" in model_name and "sem" in model_name:
        model= create_semscnn_model(input_shape)
    elif "cnn" in model_name:
        model= create_cnn_model(input_shape)
    elif "hybrid" in model_name:
        model= create_hybrid_transformer_model(input_shape)


    # x = Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
    # x = MaxPooling1D(pool_size=2, padding='same')(x)

    # x = Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    # x = MaxPooling1D(pool_size=2, padding='same')(x)

    # x = Conv1D(256, kernel_size=3, padding='same', activation='relu')(x)
    # x = MaxPooling1D(pool_size=2, padding='same')(x)

    # # Prevent further shrinking — use global pooling instead of more MaxPooling1D
    # x = GlobalAveragePooling1D()(x)

    # x = Dense(128, activation='relu')(x)
    # x = Dropout(0.5)(x)

    # if config["regression"]:
    #     outputs = Dense(1, activation='linear')(x)
    # else:
    #     outputs = Dense(1, activation='sigmoid')(x)

    # model = Model(inputs=inputs, outputs=outputs)
    return model


# if __name__ == "__main__":
#     config = {
#         "model_name": "cnn",
#         "regression": False,
#         "transformer_layers": 4,  # best 5
#         "drop_out_rate": 0.25,
#         "num_patches": 20,  # best
#         "transformer_units": 32,  # best 32
#         "regularization_weight": 0.001,  # best 0.001
#         "num_heads": 4,
#         "epochs": 100,  # best
#         "channels": [14, 18, 19, 20],
#     }
    # model = get_model(config)
    # model.build(input_shape=(1,11, 3)) # Updated shape for 11 timesteps and 3 channels
    # print(model.summary())
