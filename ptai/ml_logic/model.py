import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from colorama import Fore, Style
from typing import Tuple
from ptai.ml_logic.data import train_dataset_create, validation_dataset_create

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import convert_to_tensor



def initialize_CNN():
    """ Building of the model happens automatically just assign this function to a model variable
    and it will create a CNN Model automatically for you"""

    # initialize sequential
    model = models.Sequential()

    # Preprocessing layers
    model.add(layers.CenterCrop(height=192, width=192, input_shape=[480,480,1]))
    model.add(layers.RandomFlip(mode='horizontal', seed=123))
    # model.add(layers.RandomContrast(factor=(0.2, 0.8), seed=123))   # When we switch to rgb we can use this again

    # Build of the Model
    model.add(layers.Conv2D(filters=8, kernel_size=(4,4), activation='relu', padding='same'))
    model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Conv2D(16,(3,3), activation='relu'))
    model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Conv2D(32,(2,2), activation='relu'))
    model.add(layers.MaxPool2D((2,2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(20, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))

    print("✅ Model initialized")
    return model


def model_compile(model):
    """ Compilation of the model based on categorical crossentropy with optimizer adam """

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("✅ Model compiled")
    return model


def train_model(
        model,
        train_dataset,
        validation_dataset,
        patience=5,
        batchsize=32):
    """Train the before specified CNN Model on any images in subfolders as the labels get created automatically"""

    es = EarlyStopping(patience=patience, restore_best_weights=True)

    history = model.fit(train_dataset,
                            epochs=100,
                            validation_data=validation_dataset,
                            batch_size=batchsize,
                            callbacks=[es])
    print(f"✅ Model trained on {len(train_dataset)*32} rows with")
    return model, history



def evaluate_model(
        model,
        batch_size
        ):

    """ Evaluate the Model's performance """

    #Check if there's a model existent
    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    # commit evaluation
    metrics = model.evaluate(
        validation_dataset,
        batch_size=batch_size,
        verbose=0,
        # callbacks=None,
        return_dict=True
    )
    print(f"✅ Model evaluated")
    return metrics


def prediction_model(model):
    """Try and predict an image from the Dataset"""

    y_pred = model.predict('/home/kyrill/code/pt-ai/pt-ai/raw_data/processed_images/squats/00000003.rgb.png_squats_0.0_cropped.png')
    return y_pred

if __name__ == "__main__":
    # Creating the Data
    train_dataset = train_dataset_create()
    validation_dataset = validation_dataset_create()

    # Initializing the model
    model = initialize_CNN()
    model_compile(model)
    model.summary()

    # predicting on the model
    train_model(model, train_dataset, validation_dataset, patience=2, batchsize=16)
    evaluate_model(model, 16)
    prediction_model(model)
