# resnet_model.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Add, Flatten
from tensorflow.keras.optimizers import Adam

def ResNet(input_features=24, num_models=5):
    """
    Create a ResNet model.
    
    Parameters:
    input_features (int): Number of time series features
    num_models (int): Number of models to combine
    
    Returns:
    model: Ensemble Model
    """

    def residual_block(X, units):
        X_shortcut = X
        X = Dense(units)(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Dense(units)(X)
        X = BatchNormalization()(X)
        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)
        return X

    model = Sequential()
    model.add(Dense(64, input_dim=input_features))

    for _ in range(3):  # Add 3 residual blocks
        model.add(residual_block(model.output, 64))

    model.add(Flatten())
    model.add(Dense(num_models, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy')

    return model
