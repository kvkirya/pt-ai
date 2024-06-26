# IMPORTS

from ptai.ml_logic.data import load_image_data
from ptai.movenet.movenet_main import angle_calc
from ptai.movenet.movenet_model import load_model_from_tfhub, run_movenet_inference

from flaml import AutoML
import numpy as np
import pandas as pd
import pickle

def train_posedetect_model(X_train, y_train, settings_dict):

    """Function to train model using AutoML (FLAML)
    Be sure to specify in settings at least the following parameters:
    1) 'time_budget': <seconds, int>
    2) 'metric': <str>
    3) 'task': <classification, str>
    4) 'early_stop': <auto, str>
    5) 'estimator_list': <model type, str>

    Returns the fitted model with the best parameters as determined by AutoML
    """

    automl = AutoML()

    automl.fit(X_train=X_train, y_train=y_train, **settings_dict)

    print('Best ML model:', automl.model)
    print('Best hyperparameters:', automl.best_config)
    print('Best accuracy on validation data: {0:.4g}'.format(1 - automl.best_loss))

    best_model = automl.model

    return best_model

def posedetect_preproc(image, movenet_model):

    """This function preprocesses input for pose detection inference by:
    1) Take in any image and run skeleton detection with Movenet Single Pose Lightning
    2) Calculate the angles of the skeleton
    3) Return these angles as a dictionary"""

    keypoints = run_movenet_inference(model=movenet_model,image=image)

    keypoint_angles = angle_calc(keypoints)

    return pd.DataFrame(keypoint_angles, index=[0])

def save_posedetect_model(model, model_path):

    """Save model to specified model path"""

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model saved to: {model_path}")

def load_posedetect_model(model_path):

    """"Load posedetection model from a specified path"""

    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)

    print(f"Loaded model from: {model_path}")

    return loaded_model


def posedetect_inference(keypoints, posedetect_model, encoder = None, encoded = True):

    """Function to run inference on calculated angles of 'skeleteon' provided by Movenet
    Multiclass classifier XGBoost - returns one of three classes,
    'lunge_left': 0, 'lunge_right': 1, 'pushups': 2, 'squats': 3

    Can return classes labels as encoded or as strings
    """

    prediction = posedetect_model.predict(keypoints)

    if encoded == False:

        mapping = {0: 'lunge_left', 1: 'lunge_right', 2: 'pushup', 3: 'squat'}
        prediction_str = [mapping[i] for i in prediction]
        prediction = np.array(prediction_str)

    return prediction

if __name__ == "__main__":

    #test_img_path = input("Please enter the path to the test image")
    # test_img_path = "/Users/mischadhar/code/ds-projects/pose-detection/q4-repo/pt-ai/raw_data/test_images/IMG_1196.jpg"
    test_img_path_kyrill_pc ="/home/kyrill/code/pt-ai/pt-ai/raw_data/test_images/squat_not_chris_heria.jpg"

    test_img = load_image_data(test_img_path_kyrill_pc)

    movenet = load_model_from_tfhub()

    X_pred = posedetect_preproc(test_img, movenet)

    #model_path = input("Please enter the model path")
    # model_path = "/Users/mischadhar/code/ds-projects/pose-detection/q4-repo/pt-ai/raw_data/models/best_automl_model.pkl"
    model_path_kyrill_pc = "/home/kyrill/code/pt-ai/pt-ai/raw_data/models/best_automl_model.pkl"
    loaded_model = load_posedetect_model(model_path_kyrill_pc)

    y_pred = posedetect_inference(X_pred, loaded_model, encoder=None, encoded=False)

    print(y_pred)
