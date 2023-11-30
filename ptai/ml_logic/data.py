import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Needed for the model filtering
from tensorflow.keras.utils import image_dataset_from_directory

#params
batch_size = 32
img_height = 256
img_width = 256
validation_split = 0.2
num_classes = 4

def train_dataset_create():
    train_dataset = image_dataset_from_directory(
        directory='/home/kyrill/code/pt-ai/pt-ai/raw_data/processed_data_03',
        labels='inferred',
        label_mode='categorical',
        color_mode='grayscale',
        validation_split=validation_split,
        subset='training',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    return train_dataset


def validation_dataset_create():
    validation_dataset = image_dataset_from_directory(
    directory='/home/kyrill/code/pt-ai/pt-ai/raw_data/processed_data_03',
    labels='inferred',
    label_mode='categorical',
    color_mode='grayscale',
    validation_split=validation_split,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
    return validation_dataset
