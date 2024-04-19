import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D
from tensorflow.keras.models import Model

def CustomModel():
    # Input layer
    input_layer = Input(shape=(64, 64, 3))  # Input image size is 64x64x3
    
    # Convolutional layers
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    
    # Upsampling layers
    x = UpSampling2D((4, 4))(x)  # Upsampling factor to produce 256x256 output
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    
    x = UpSampling2D((4, 4))(x)  # Upsampling factor to produce 256x256 output
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    
    # Output layer
    output_layer = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model
