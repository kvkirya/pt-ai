from PIL import Image
from tensorflow import convert_to_tensor
import tensorflow as tf
import numpy as np
from tensorflow.image import rgb_to_grayscale

def preprocessing_images(image_array):
    X_pred = tf.image.rot90(image_array, k=-1)

    X_pred = X_pred[:,:,:3]/255.
    X_pred = tf.image.resize_with_pad(X_pred, 256,256)
    X_pred = rgb_to_grayscale(X_pred)

    X_pred = tf.expand_dims(X_pred, axis=0)

    # plt.imshow(rgb_to_grayscale(X_pred[:,:,:3]/255.), cmap='gray')
    return X_pred
