import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.feature import hog
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model

def extract_handcrafted_features(img):
    """
    Extract GLCM, HOG, and LBP features from an image
    """
    # GLCM
    glcm = graycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
    contrast = graycoprops(glcm, 'contrast').mean()
    
    # HOG
    hog_features = hog(img, pixels_per_cell=(16,16))
    
    # LBP
    lbp_features = local_binary_pattern(img, P=8, R=1).flatten()
    
    return np.concatenate([[contrast], hog_features, lbp_features])

def extract_deep_features(img):
    """
    Extract deep features using pretrained ResNet50
    """
    img_rgb = np.stack((img,)*3, axis=-1)  # convert grayscale to RGB
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
    model = Model(inputs=base_model.input, outputs=base_model.output)
    features = model.predict(np.expand_dims(img_rgb, axis=0))
    return features.flatten()
