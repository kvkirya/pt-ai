from ptai.ml_logic.preprocessor import preprocessing_images_vgg16

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model as tfk__load_model



def initialize_CNN():
    """ Building of the model happens automatically just assign this function to a model variable
    and it will create a CNN Model automatically for you"""

    # initialize sequential
    model = models.Sequential()

    # Preprocessing layers
    model.add(layers.CenterCrop(height=192, width=192, input_shape=[256,256,1]))
    model.add(layers.RandomFlip(mode='horizontal', seed=123))
    # model.add(layers.RandomContrast(factor=(0.2, 0.8), seed=123))   # When we switch to rgb we can use this again

    # Build of the Model
    model.add(layers.Conv2D(filters=16, kernel_size=(4,4), activation='relu', padding='same'))
    model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Conv2D(32,(3,3), activation='relu'))
    model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Conv2D(64,(2,2), activation='relu'))
    model.add(layers.MaxPool2D((2,2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(25, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(14, activation='relu'))
    model.add(layers.Dense(10, activation='relu'))
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
                            epochs=500,
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
    print(metrics)
    return metrics

def load_model(path_to_model):
    fitted_model = tfk__load_model(path_to_model)
    return fitted_model

def prediction_model(model, image_to_pass):
    """Try and predict an image from the Dataset"""
    image_to_pred = preprocessing_images_vgg16(image_to_pass)
    y_pred = model.predict(image_to_pred)
    print(f'✅ Prediction complete. Pose: {y_pred}')
    return y_pred

if __name__ == "__main__":
    # Creating the Data
    """
    current_wd = os.getcwd()
    path_to_raw_data = os.path.join(current_wd,"../../raw_data")
    train_dataset = train_dataset_create()
    validation_dataset = validation_dataset_create()
    """
    # Initializing the model
    """
    model = initialize_CNN()
    model_compile(model)
    model.summary()
    """
    # predicting on the model
    """
    train_model(model, train_dataset, validation_dataset, patience=10, batchsize=128)
    model.save('/home/jupyter/pt-ai/raw_data/models/model.h5')
    """

    # The true prediction
    fitted_model = tfk__load_model('../../raw_data/models/model.h5')
    prediction_model(fitted_model, '../../raw_data/test_images/IMG_8805.jpg')

    #model.save(os.path.join(path_to_raw_data,"models/model.h5"))
    # evaluate_model(model, 16)
