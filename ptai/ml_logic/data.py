import os
import json

from ptai.data_preproc import crop_images

def generate_cropped_images(path_to_raw_data):

    with open(os.path.join(path_to_raw_data, "annotations_cleaned_IoU.json"), 'r') as file:
        data = json.load(file)

    data_no_overlap = [annotation for annotation in data if annotation["iou_cat"] == False]

    for annotation in data_no_overlap:
        image_id = annotation["image_id"]
        bbox = annotation["bbox"]
        iou = sum(annotation["iou"])
        pose_variation = annotation["pose_variation"]

        image_id_len = len(str(image_id))
        zeros = "0" * (8 - image_id_len)

        file_name = f"{zeros}{image_id}.rgb.png"

        cropped_image = crop_images.crop_image_to_bounding_box(file_name, os.path.join(path_to_raw_data,"00"), bbox)

        pose_subdirs = os.listdir(os.path.join(path_to_raw_data, "cropped_data"))

        if pose_variation not in pose_subdirs:
            os.mkdir(os.path.join(path_to_raw_data, "cropped_data", pose_variation))

        crop_images.save_cropped_image(cropped_image
                                    , os.path.join(path_to_raw_data, "cropped_data", pose_variation)
                                    , file_name + "_" + pose_variation + "_" + str(round(iou,3)) + "_cropped.png")

if __name__ == "__main__":

    path_to_raw_data = "raw_data"

    generate_cropped_images(path_to_raw_data)
