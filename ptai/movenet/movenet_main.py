# IMPORTS
import matplotlib.pyplot as plt
import cv2
# import mediapipe as mp
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from ptai.movenet.movenet_model import load_model_from_tfhub, run_movenet_inference
from ptai.movenet.movenet_data import load_image_data, load_image_for_skeleton
from ptai.movenet.movenet_plot_img import draw_prediction_on_image

#Works with docker
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


# Function to calculate the 3D angles
def angle_triangle(x1, y1, z1, x2, y2, z2, x3, y3, z3):

    BAx = x1-x2
    BAy = y1-y2
    BAz = z1-z2

    BCx = x3-x2
    BCy = y3-y2
    BCz = z3-z2

    num = BAx*BCx+BAy*BCy+BAz*BCz

    den = (np.sqrt(BAx**2+BAy**2+BAz**2))*\
                (np.sqrt((BCx)**2+(BCy)**2+(BCz)**2))

    angle = np.degrees(np.arccos(num / den))

    return round(angle, 3)

# Application on the function on the points
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End

    angle = angle_triangle(*a, *b, *c) # use the angle_triangle function

    if angle >180.0:
        angle = 360-angle

    return angle

#The following dictionary maps body parts to keypoint indices
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Maps the edges of the skeleton to colours
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'g',
    (0, 2): 'g',
    (1, 3): 'g',
    (2, 4): 'g',
    (0, 5): 'g',
    (0, 6): 'g',
    (5, 7): 'g',
    (7, 9): 'g',
    (6, 8): 'g',
    (8, 10): 'g',
    (5, 6): 'g',
    (5, 11): 'g',
    (6, 12): 'g',
    (11, 12): 'g',
    (11, 13): 'g',
    (13, 15): 'g',
    (12, 14): 'g',
    (14, 16): 'g',
}

#For squats, we want shoulder, hip, knee, ankle,
Squat = {
    "left_elbow_a": 0,
    "right_elbow_a": 0,
    "left_shoulder_b": 0,
    "left_shoulder_a": 1,
    "right_shoulder_b": 0,
    "right_shoulder_a": 0,
    "left_elbow": 0,
    "right_elbow": 0,
    "left_wrist": 0,
    "right_wrist": 0,
    "left_hip_a": 0,
    "left_hip_b": 1, #we want angle between these three
    "left_hip_c": 0,
    "right_hip_a": 0,
    "right_hip_b": 0,
    "right_hip_c": 1, #need this one too
    "left_knee_a": 1,
    "right_knee_a": 1, #we want angle between these three
    "left_ankle": 1,
    "right_ankle": 0
}

Squat_angles_ideal = {
    "left_hip_b": 90, #we want angle between these three
    "right_hip_c": 90, #need this one too
    "left_knee_a": 90,
    "right_knee_a": 90, #we want angle between these three
}


#For Pushups, we want wrist, elbow, shoulder, hip, knee
Pushup = {
    "left_elbow_a": 1,
    "right_elbow_a": 1,
    "left_shoulder_b": 0,
    "left_shoulder_a": 1,
    "right_shoulder_b": 0,
    "right_shoulder_a": 1,
    "left_elbow": 0,
    "right_elbow": 0,
    "left_wrist": 0,
    "right_wrist": 0,
    "left_hip_a": 1,
    "left_hip_b": 0,
    "left_hip_c": 0,
    "right_hip_a": 0,
    "right_hip_b": 0,
    "right_hip_c": 0,
    "left_knee_a": 1,
    "right_knee_a": 1,
    "left_ankle": 0,
    "right_ankle": 0
}

Pushup_angles_ideal = {
    "left_elbow_a": 90,
    "left_shoulder_a": 90,
    "left_hip_a": 180,
    "left_knee_a": 180,
    "right_knee_a": 180,
    "right_elbow_a": 90,
    "right_shoulder_a": 90
}

#For Lunges, we want hip, knee, foot
Lunge = {
    "left_elbow_a": 0,
    "right_elbow_a": 0,
    "left_shoulder_b": 0,
    "left_shoulder_a": 0,
    "right_shoulder_b": 0,
    "right_shoulder_a": 0,
    "left_elbow": 0,
    "right_elbow": 0,
    "left_wrist": 0,
    "right_wrist": 0,
    "left_hip_a": 1,
    "left_hip_b": 0,
    "left_hip_c": 0,
    "right_hip_a": 0,
    "right_hip_b": 1,
    "right_hip_c": 0,
    "left_knee_a": 1,
    "right_knee_a": 0,
    "left_ankle": 1,
    "right_ankle": 0
}

Lunge_left_angles_ideal = {
    "left_hip_a": 90,
    "left_knee_a": 90,
    "right_knee_a": 90,
    "right_hip_a": 180
}

Lunge_right_angles_ideal = {
    "left_hip_a": 180,
    "left_knee_a": 90,
    "right_knee_a": 90,
    "right_hip_a": 90
}

def angle_calc(keypoints_with_scores):
    """applying he calculate angles function on all angles defined in the angles_dictionary"""

    key_xy = keypoints_with_scores[:, :, :, :3] #pick what body parts we take for the angles

    # Create a dictionary of keypoints and their corresponding vector tuples
    key_dict = {}
    for key, value in KEYPOINT_DICT.items():
        vector = tuple(key_xy[0, 0, value])
        key_dict[key] = vector

    angles_dictionary = {
    "left_elbow_a": (key_dict["left_wrist"],key_dict["left_elbow"],key_dict["left_shoulder"]),
    "right_elbow_a": (key_dict["right_wrist"],key_dict["right_elbow"],key_dict["right_shoulder"]),
    "left_shoulder_b": (key_dict["left_elbow"],key_dict["left_shoulder"],key_dict["left_hip"]),
    "left_shoulder_a": (key_dict["left_hip"],key_dict["left_shoulder"],key_dict["right_shoulder"]),
    "right_shoulder_b": (key_dict["right_elbow"],key_dict["right_shoulder"],key_dict[ "right_hip"]),
    "right_shoulder_a": (key_dict["right_hip"],key_dict["right_shoulder"],key_dict[ "left_shoulder"]),
    "left_hip_a": (key_dict["left_shoulder"],key_dict["left_hip"],key_dict[ "right_hip"]),
    "left_hip_b": (key_dict["left_shoulder"],key_dict["left_hip"],key_dict[ "left_knee"]),
    "left_hip_c": (key_dict["right_hip"],key_dict["left_hip"],key_dict[ "left_knee"]),
    "right_hip_a": (key_dict["right_shoulder"],key_dict["right_hip"],key_dict[ "left_hip"]),
    "right_hip_b": (key_dict["left_hip"],key_dict["right_hip"],key_dict[ "right_knee"]),
    "right_hip_c": (key_dict["right_shoulder"],key_dict["right_hip"],key_dict[ "right_knee"]),
    "left_knee_a": (key_dict["left_hip"],key_dict["left_knee"],key_dict[ "left_ankle"]),
    "right_knee_a": (key_dict["right_hip"],key_dict["right_knee"],key_dict[ "right_ankle"])
    }

    angles = {}
    for key, value in angles_dictionary.items():
        angle = calculate_angle(value[0], value[1], value[2]) #calculates the angle between the 3
        angles[key] = angle
    return angles


#give this function the angles array calculated above and the ideal angles defined for the movements
def compare_angles(angles, ideal_angles):
    """angles is the predicted values by the machine and ideal_angles are the
    angles considered ideal by us in any movement."""
    dict1 = angles
    dict2 = ideal_angles
    flexed_dict = {}

    for dict1_key in dict1:
        flexed_dict[dict1_key] = abs(dict2[dict1_key] - dict1[dict1_key])
    return flexed_dict

def render_red(flexed_dict, KEYPOINT_EDGE_INDS_TO_COLOR, threshold=10):
    '''
    Takes the dict of differences in angles and looks for an angle difference more than 10 degrees,
    then changes the colours of the corresponding bars around the angle to red.
    '''

    #indicate what points form an edge
    bars_dictionary = {
        "left_elbow_a": [(5,7),(7,9)],
        "right_elbow_a": [(6,8),(8,10)],
        "left_shoulder_b": [(5,7),(5,11)],
        "left_shoulder_a": [(5,6),(5,11)],
        "right_shoulder_b": [(6,8),(6,12)],
        "right_shoulder_a": [(5,6),(6,12)],
        "left_hip_a": [(5,11),(11,12)],
        "left_hip_b": [(11,12),(11,13)],
        "left_hip_c": [(5,11),(11,13)],
        "right_hip_a": [(6,12),(11,12)],
        "right_hip_b": [(11,12),(12,14)],
        "right_hip_c": [(6,12),(12,14)],
        "left_knee_a": [(11,13),(13,15)],
        "right_knee_a": [(12,14),(14,16)]
    }
    RED_EDGES = KEYPOINT_EDGE_INDS_TO_COLOR.copy()

    # check for difference if larger then 7 then mark as red --> 7 degree threshold
    for k, v in flexed_dict.items():
        if v >= threshold:
            points_to_color = bars_dictionary[k]
            for point in points_to_color:
                RED_EDGES[point] = 'r'
    return RED_EDGES


def _keypoints_and_edges_for_display_red(keypoints_with_scores, RED_EDGES, height, width, keypoint_threshold=0.11):
    """Returns high confidence keypoints and edges for visualization.
    Args:
        keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
        the keypoint coordinates and scores returned from the MoveNet model.
        height: height of the image in pixels.
        width: width of the image in pixels.
        keypoint_threshold: minimum confidence score for a keypoint to be
        visualized.

    Returns:
        A (keypoints_xy, edges_xy, edge_colors) containing:
        * the coordinates of all keypoints of all detected entities;
        * the coordinates of all skeleton edges of all detected entities;
        * the colors in which the edges should be plotted.
    """

    keypoints_all = []
    keypoint_edges_all = []
    edge_colors = []
    num_instances, _, _, _ = keypoints_with_scores.shape
    for idx in range(num_instances):
        kpts_x = keypoints_with_scores[0, idx, :, 1]
        kpts_y = keypoints_with_scores[0, idx, :, 0]
        kpts_scores = keypoints_with_scores[0, idx, :, 2]
        kpts_absolute_xy = np.stack(
            [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
        kpts_above_thresh_absolute = kpts_absolute_xy[
            kpts_scores > keypoint_threshold, :]
        keypoints_all.append(kpts_above_thresh_absolute)

    for edge_pair, color in RED_EDGES.items():
        if (kpts_scores[edge_pair[0]] > keypoint_threshold and
            kpts_scores[edge_pair[1]] > keypoint_threshold):
            x_start = kpts_absolute_xy[edge_pair[0], 0]
            y_start = kpts_absolute_xy[edge_pair[0], 1]
            x_end = kpts_absolute_xy[edge_pair[1], 0]
            y_end = kpts_absolute_xy[edge_pair[1], 1]
            line_seg = np.array([[x_start, y_start], [x_end, y_end]])
            keypoint_edges_all.append(line_seg)
            edge_colors.append(color)
    if keypoints_all:
        keypoints_xy = np.concatenate(keypoints_all, axis=0)
    else:
        keypoints_xy = np.zeros((0, 17, 2))

    if keypoint_edges_all:
        edges_xy = np.stack(keypoint_edges_all, axis=0)
    else:
        edges_xy = np.zeros((0, 2, 2))
    return keypoints_xy, edges_xy, edge_colors

def draw_prediction_on_image_red(image, keypoints_with_scores, red_edges, crop_region=None, close_figure=False, output_image_height=None):
    """Draws the keypoint predictions on image.
    Args:
        image: A numpy array with shape [height, width, channel] representing the
        pixel values of the input image.
        keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
        the keypoint coordinates and scores returned from the MoveNet model.
        crop_region: A dictionary that defines the coordinates of the bounding box
        of the crop region in normalized coordinates (see the init_crop_region
        function below for more detail). If provided, this function will also
        draw the bounding box on the image.
        output_image_height: An integer indicating the height of the output image.
        Note that the image aspect ratio will be the same as the input image.

    Returns:
        A numpy array with shape [out_height, out_width, channel] representing the
        image overlaid with keypoint predictions.
    """

    height, width, channel = image.shape
    aspect_ratio = float(width) / height
    fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
    # To remove the huge white borders
    fig.tight_layout(pad=0)
    ax.margins(0)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.axis('off')

    im = ax.imshow(image)
    line_segments = LineCollection([], linewidths=(4), linestyle='solid')
    ax.add_collection(line_segments)
    # Turn off tick labels
    scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)

    (keypoint_locs, keypoint_edges,
    edge_colors) = _keypoints_and_edges_for_display_red(keypoints_with_scores, red_edges,
                                                    height=height,
                                                    width=width,
                                                    keypoint_threshold=0.11)

    line_segments.set_segments(keypoint_edges)
    line_segments.set_color(edge_colors)
    if keypoint_edges.shape[0]:
        line_segments.set_segments(keypoint_edges)
        line_segments.set_color(edge_colors)
    if keypoint_locs.shape[0]:
        scat.set_offsets(keypoint_locs)

    if crop_region is not None:
        xmin = max(crop_region['x_min'] * width, 0.0)
        ymin = max(crop_region['y_min'] * height, 0.0)
        rec_width = min(crop_region['x_max'], 0.99) * width - xmin
        rec_height = min(crop_region['y_max'], 0.99) * height - ymin
        rect = patches.Rectangle(
            (xmin,ymin),rec_width,rec_height,
            linewidth=1,edgecolor='b',facecolor='none')
        ax.add_patch(rect)

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(
        fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    #if output_image_height is not None:
     #   output_image_width = int(output_image_height / height * width)
      #  image_from_plot = cv2.resize(
       #     image_from_plot, dsize=(output_image_width, output_image_height),
        #    interpolation=cv2.INTER_CUBIC)
    return image_from_plot


def plot_red(keypoints_with_scores,image, red_edges):
    if image.shape[2] == 4:
            image = image[:,:,:3]

    display_image = tf.expand_dims(image, axis=0)
    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    display_image = tf.cast(tf.image.resize_with_pad(image, 192, 192), dtype=tf.int32)

    output_overlay = draw_prediction_on_image_red(
        display_image, keypoints_with_scores, red_edges)

    # plt.figure(figsize=(5, 5))
    # plt.imshow(output_overlay)
    # _ = plt.axis('off')
    return output_overlay


if __name__ == "__main__":

 #   image_path = '/home/kyrill/code/pt-ai/pt-ai/raw_data/test_images/IMG_8806.jpg'
 #   image = tf.keras.utils.load_img(image_path)
 #   input_arr = tf.keras.utils.img_to_array(image)
 #   input_arr = np.array([input_arr])
 #   print(input_arr.shape)

    # response = requests.post("http://localhost:8080/skeletonizer", json=json.dumps(input_arr.tolist()))
    # keypoints_with_scores = response.json()
    # keypoints_with_scores = eval(keypoints_with_scores)
    # keypoints_with_scores = np.array(keypoints_with_scores)
    # # keypoints_with_scores = load_model_and_run_inference(image_path=image_path)


    # plot_skeleton_on_image(image_path, keypoints_with_scores)

    image_path_1 = "/home/kyrill/code/pt-ai/pt-ai/raw_data/test_images/IMG_8806.jpg"
    image_path_2 = '/home/kyrill/code/pt-ai/pt-ai/raw_data/test_images/woman_pushup.jpg'

    keypoints_with_scores_im1 = load_model_and_run_inference(image_path=image_path_1)
    print(keypoints_with_scores_im1)
    keypoints_with_scores_im2 = load_model_and_run_inference(image_path=image_path_2)

    angles = angle_calc(keypoints_with_scores_im1)
    prediction = angle_calc(keypoints_with_scores_im2)
    threshold = 20
    flexed_dict = compare_angles(prediction, angles)

    KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'g',
    (0, 2): 'g',
    (1, 3): 'g',
    (2, 4): 'g',
    (0, 5): 'g',
    (0, 6): 'g',
    (5, 7): 'g',
    (7, 9): 'g',
    (6, 8): 'g',
    (8, 10): 'g',
    (5, 6): 'g',
    (5, 11): 'g',
    (6, 12): 'g',
    (11, 12): 'g',
    (11, 13): 'g',
    (13, 15): 'g',
    (12, 14): 'g',
    (14, 16): 'g',
}

    colored_edges = render_red(flexed_dict, KEYPOINT_EDGE_INDS_TO_COLOR, threshold)

    image = tf.io.read_file(image_path_2)
    image = tf.image.decode_png(image)

    height=192
    width=192

    print(plot_red(keypoints_with_scores_im1, image, colored_edges))

    #print(draw_prediction_on_image(image,keypoints_with_scores_im2))
    #print(_angles(prediction, angles))
    #print(keypoints_with_scores.shape)
    # plot_skeleton_on_image(image_path, keypoints_with_scores)


# -------------------------------------------------------------
# compare_angles function to
# return a boolean value for each angle based on a threshold.
# ff the angle is less than threshold, consider flexed;
# otherwise not flexed

# aka ....dict < threshold which is inputed as threshold=90

#  flexed_dict = {}

#     for dict1_key, dict1_values in dict1.items():
#         flexed_dict[dict1_key] = abs(dict2[dict1_key] - dict1[dict1_key]) < threshold

#     return flexed_dict



# angles = angle_calc(keypoints_with_scores_im1)
# prediction = angle_calc(keypoints_with_scores_im2)

# flexed_dict = compare_angles(prediction, angles, threshold)
