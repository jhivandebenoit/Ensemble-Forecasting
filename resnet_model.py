# resnet_model.py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, Add, GlobalAveragePooling1D, Dense, Softmax

def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    
    x = Conv1D(filters=filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv1D(filters=filters, kernel_size=kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    x = Add()([x, shortcut])
    x = ReLU()(x)
    
    return x

def Resnet_META(input_shape=(788, 1), num_classes=4):
    inputs = Input(shape=input_shape)
    
    x = Conv1D(filters=64, kernel_size=7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = residual_block(x, filters=64)
    x = residual_block(x, filters=128, stride=2)
    x = residual_block(x, filters=256, stride=2)
    
    x = GlobalAveragePooling1D()(x)
    x = Dense(units=num_classes)(x)
    outputs = Softmax()(x)

    model = Model(inputs=inputs, outputs=outputs)
    
    return model
