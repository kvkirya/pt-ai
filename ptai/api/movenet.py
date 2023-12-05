from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from ptai.movenet.movenet_model import load_model_from_tfhub, run_movenet_inference
import numpy as np
import tensorflow as tf
import json

app = FastAPI()

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app = FastAPI()


model = load_model_from_tfhub()


@app.post("/skeletonizer")
async def get_skeletonization(input_arr: Request):
    data = await input_arr.json()
    data = np.array(json.loads(data))

    image = tf.convert_to_tensor(data)


    # image = tf.expand_dims(image, axis=0)
        # Resize and pad the image to keep the aspect ratio and fit the expected size.
    image = tf.cast(tf.image.resize_with_pad(image, 192, 192), dtype=tf.int32)


    keypoints_with_scores = run_movenet_inference(model, image)

    return keypoints_with_scores.tolist().__repr__()
