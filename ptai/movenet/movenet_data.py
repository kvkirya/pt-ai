import tensorflow as tf

def load_image_data(image_path, width=192, height=192, source="LOCAL"):
    """This returns the image in the right dimensions that is then used
    to plot the points and run the movenet
    """
    if source == "LOCAL":

        image = tf.io.read_file(image_path)
        image = tf.compat.v1.image.decode_png(image)

        if image.shape[2] == 4:
            image = image[:,:,:3]

        image = tf.expand_dims(image, axis=0)
        # Resize and pad the image to keep the aspect ratio and fit the expected size.
        image_resized = tf.image.resize_with_pad(image, width, height, antialias = True)
        image = tf.cast(image_resized, dtype=tf.int32)

    return image

def load_image_for_skeleton(image_path):

    image = tf.io.read_file(image_path)

    image = tf.image.decode_png(image)

    if image.shape[2] == 4:

        image = image[:,:,:3]

    display_image = tf.expand_dims(image, axis=0)
    display_image = tf.cast(tf.image.resize_with_pad(display_image, 1280, 1280)
                            , dtype=tf.int32)

    return display_image
