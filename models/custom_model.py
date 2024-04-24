import tensorflow as tf
from tensorflow.keras import layers, models

def custom_model(input_shape=(None, None, 3)):
    # Input layer
    inputs = layers.Input(shape=input_shape)

    # Initial Convolutional Layer
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)

    # Residual blocks
    for _ in range(16):
        x = residual_block(x)

    # Upsampling blocks
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D(size=(2, 2))(x)

    # Final Convolutional Layer
    outputs = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    model = models.Model(inputs, outputs)
    return model

def residual_block(x):
    y = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    y = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(y)
    return layers.Add()([x, y])

# Create the ESRGAN generator model
esrgan_gen_model = custom_model()
esrgan_gen_model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Display the model summary
esrgan_gen_model.summary()
