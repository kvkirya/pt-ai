import os
import json
from tqdm import tqdm
import json

# Needed for the model filtering
from tensorflow.keras.utils import image_dataset_from_directory

from ptai.data_preproc import crop_images

from ptai.movenet.movenet_data import load_image_data
from ptai.movenet.movenet_model import load_model_from_tfhub, run_movenet_inference

def generate_cropped_images(path_to_raw_data):

    with open(os.path.join(path_to_raw_data, "../annotations_cleaned_IoU.json"), 'r') as file:
        data = json.load(file)

    data_no_overlap = [annotation for annotation in data if annotation["iou_cat"] == False]

    for annotation in tqdm(data_no_overlap):
        image_id = annotation["image_id"]
        bbox = annotation["bbox"]
        iou = sum(annotation["iou"])
        pose_variation = annotation["pose_variation"]

        if pose_variation in ['lunge_left', 'lunge_right', 'squats', 'pushups']:

            image_id_len = len(str(image_id))
            zeros = "0" * (8 - image_id_len)

            file_name = f"{zeros}{image_id}.rgb.png"

            cropped_image = crop_images.crop_image_to_bounding_box(file_name, os.path.join(path_to_raw_data), bbox)

            pose_subdirs = os.listdir(os.path.join(path_to_raw_data, "../cropped_data"))

            if pose_variation not in pose_subdirs:
                os.mkdir(os.path.join(path_to_raw_data, "../cropped_data", pose_variation))

            crop_images.save_cropped_image(cropped_image
                                        , os.path.join(path_to_raw_data, "../cropped_data", pose_variation)
                                        , file_name.removesuffix(".rgb.png") + "_" + pose_variation + "_" + str(round(iou,3)) + "_cropped.png")

def generate_keypoints_for_xgboost(path_to_cropped_images):

    keypoints_list = []
    pose_list = []

    model = load_model_from_tfhub()

    pose_folders = os.listdir(path_to_cropped_images)
    pose_folders.remove(".DS_Store")

    for folder in pose_folders:

        folder_path = os.path.join(path_to_cropped_images,folder)

        images = os.listdir(folder_path)

        for image in tqdm(images):

            img_path = os.path.join(os.path.join(path_to_cropped_images,folder),image)

            img_array = load_image_data(img_path)

            keypoints = run_movenet_inference(model, img_array)

            keypoints_list.append(keypoints)
            pose_list.append(folder)

    return keypoints_list, pose_list

#params
batch_size = 32
img_height = 256
img_width = 256
validation_split = 0.2
num_classes = 4

def train_dataset_create():
    train_dataset = image_dataset_from_directory(
        directory='/home/kyrill/code/pt-ai/pt-ai/raw_data/processed_images',
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
        directory='/home/kyrill/code/pt-ai/pt-ai/raw_data/processed_images',
        labels='inferred',
        label_mode='categorical',
        color_mode='grayscale',
        validation_split=validation_split,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    return validation_dataset

if __name__ == "__main__":

    # path_to_raw_data = "raw_data"
    # unprocessed_data_folder = os.path.join(path_to_raw_data,"unprocessed_data")

    # generate_cropped_images(unprocessed_data_folder)

    current_wd = os.getcwd()

    path_to_cropped_images = os.path.join(current_wd,"raw_data/cropped_data")

    X, y = generate_keypoints_for_xgboost(path_to_cropped_images)

    X_filename = os.path.join(current_wd,'raw_data/X_list.json')
    y_filename = os.path.join(current_wd,'raw_data/y_list.json')

    with open(X_filename, 'w') as file:
        # Serialize and save the list
        json.dump(X, file)

    print("Saved X file")

    with open(y_filename, 'w') as file:
        # Serialize and save the list
        json.dump(y, file)

    print("Saved y file")
