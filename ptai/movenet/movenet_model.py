import tensorflow_hub as hub

def load_model_from_tfhub(model_url="""https://www.kaggle.com/models/google/movenet/frameworks/TensorFlow2/variations/singlepose-lightning/versions/4"""):

    """Load a tfhub model from a given URL"""

    model = hub.load(model_url)

    movenet = model.signatures['serving_default']

    return movenet

def run_movenet_inference(model, image):
    # Run model inference.
    outputs = model(image)

    # Output is a [1, 1, 17, 3] tensor.
    keypoints_with_scores = outputs['output_0'].numpy()

    return keypoints_with_scores
