import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_image(img, target_size=(224,224)):
    """
    Resize, normalize and return image
    """
    img_resized = cv2.resize(img, target_size)
    img_norm = (img_resized - np.mean(img_resized)) / np.std(img_resized)
    return img_norm

def augment_images(images):
    """
    Apply data augmentation: rotation, flip, zoom
    """
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )
    return datagen.flow(images, batch_size=len(images))
