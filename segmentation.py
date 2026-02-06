import tensorflow as tf
from tensorflow.keras import layers, models

def build_crp_urnet(input_shape=(224,224,3)):
    """
    Simplified CRP-URNet architecture for segmentation
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    # Example encoder
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    
    # Example decoder
    x = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
