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

train_dataset = train_dataset_create()
validation_dataset = validation_dataset_create()

def initialize_CNN():
    model = models.Sequential()

    # Preprocessing layers
    model.add(layers.CenterCrop(height=350, width=450, input_shape=[256,256,1]))

    # Build of the Model
    model.add(layers.Conv2D(filters=8, kernel_size=(4,4), activation='relu', padding='same'))
    model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Conv2D(16,(3,3), activation='relu'))
    model.add(layers.MaxPool2D((2,2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))

    print("✅ Model initialized")
    # Compilation of the Model
    return model


def model_compile(model:initialize_CNN()):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("✅ Model compiled")
    return model


def train_model(
        model:model_compile,
        train_dataset,
        patience=5,
        validation_dataset=validation_dataset
        ):

    es = EarlyStopping(patience=patience, restore_best_weights=True)

    history = model_compile.fit(train_dataset,
                            epochs=100,
                            validation_data=validation_dataset,
                            batch_size=16,
                            callbacks=[es],
                            vebose=0)
    print(f"✅ Model trained on {len(train_dataset)*32} rows with")
    return model, history


#add model_evaluate
def evaluate_model(
        model:model_compile
        ):
    pass
