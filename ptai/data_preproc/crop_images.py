import matplotlib.pyplot as plt
import os
from PIL import Image

def crop_image_to_bounding_box(file_name, raw_image_path, bbox, padding=15):
    """
    Crops an image to the specified bounding box with additional padding.
    """
    # Open the image
    image = Image.open(os.path.join(raw_image_path, file_name))

    # Extract the dimensions of the image
    img_width, img_height = image.size

    # COCO bbox format is [x_min, y_min, width, height]
    x_min, y_min, width, height = bbox

    # Calculate the coordinates with padding
    x_min_pad = max(0, x_min - padding)
    y_min_pad = max(0, y_min - padding)
    x_max_pad = min(img_width, x_min + width + padding)
    y_max_pad = min(img_height, y_min + height + padding)

    # Crop the image
    cropped_image = image.crop((x_min_pad, y_min_pad, x_max_pad, y_max_pad))

    return cropped_image

def save_cropped_image(cropped_image, cropped_image_path, cropped_image_name):
    """
    Saves the cropped image to the specified path.
    """
    cropped_image.save(os.path.join(cropped_image_path,cropped_image_name))
