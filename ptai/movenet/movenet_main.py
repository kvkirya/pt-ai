# IMPORTS
import matplotlib.pyplot as plt
import numpy as np
import os

from ptai.movenet.movenet_model import load_model_from_tfhub, run_movenet_inference
from ptai.movenet.movenet_data import load_image_data, load_image_for_skeleton
from ptai.movenet.movenet_plot_img import draw_prediction_on_image

import requests
import numpy as np
import json
import tensorflow as tf
def load_model_and_run_inference(image_path):

    """This function:
    1) Loads the model from tfhub (default is single-pose lightning)
    2) Loads the image to run inference on either locally or from cloud
    (to be coded) - default is LOCAL
    3) Runs inference on the image and returns the keypoints of the skeleton"""

    model = load_model_from_tfhub()

    image = load_image_data(image_path)

    keypoints_with_scores = run_movenet_inference(model, image)

    return keypoints_with_scores

def plot_skeleton_on_image(image_path, keypoints_with_scores):

    display_image = load_image_for_skeleton(image_path)

    output_overlay = draw_prediction_on_image(
        np.squeeze(display_image.numpy(), axis=0), keypoints_with_scores)


    plt.figure(figsize=(5, 5))
    plt.imshow(output_overlay)
    _ = plt.axis('off')
    plt.show()

if __name__ == "__main__":


    image_path = "/Users/nicowsendagorta/code/kvkirya/pt-ai/raw_data/test_squat.png"
    image = tf.keras.utils.load_img(image_path)
    input_arr = tf.keras.utils.img_to_array(image)
    input_arr = np.array([input_arr])
    print(input_arr.shape)


    response = requests.post("http://localhost:8080/skeletonizer", json=json.dumps(input_arr.tolist()))
    keypoints_with_scores = response.json()
    keypoints_with_scores = eval(keypoints_with_scores)
    keypoints_with_scores = np.array(keypoints_with_scores)
    # keypoints_with_scores = load_model_and_run_inference(image_path=image_path)


    plot_skeleton_on_image(image_path, keypoints_with_scores)






# response = requests.post("https://app-nkoiw7qw6a-ew.a.run.app/dummy", json=json.dumps(test.tolist()))
# print(response)
