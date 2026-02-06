import tensorflow as tf
from tensorflow.keras import layers, models

def build_widenet(input_dim):
    """
    WideNet fully connected classifier
    """
    inputs = tf.keras.Input(shape=(input_dim,))
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(2, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
