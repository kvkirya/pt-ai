from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from tensorflow import convert_to_tensor
from ptai.ml_logic.model import load_model, prediction_model
from ptai.movenet.movenet_model import load_model_from_tfhub, run_movenet_inference
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model as tfk__load_model
from PIL import Image
from fastapi import FastAPI, File, UploadFile

import tensorflow as tf
import json
import io
import numpy as np


# Create the app so fastapi knows what to execute
app = FastAPI()

#no clue
origins = ['*']

#no clue
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Add the CNN model
model_CNN = tfk__load_model('/ptai-proj/raw_data/models/model_1.h5', compile=False)
model_CNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
app.state.model = model_CNN

# find the movenet model online and load it
model_movenet = load_model_from_tfhub()

# Create the first endpoint for the CNN
@app.post("/uploadfile/")
async def prediction(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    img_array = convert_to_tensor(np.array(image), dtype=tf.float32)

    model = app.state.model
    assert model is not None
    y_pred = prediction_model(model, img_array)
    return JSONResponse(content = {'pose':f'Predicted pose is: {y_pred}'})


# Create the second endpoint for the movenet model
@app.post("/skeletonizer/")
async def get_skeletonization(file: UploadFile = File(...)):
    data = await file.read()
    data = Image.open(io.BytesIO(data))

    # turn the image into a tensor
    image = convert_to_tensor(np.array(data), dtype=tf.float32)
    image = tf.cast(tf.image.resize_with_pad(image, 192, 192), dtype=tf.int32)
    image = tf.expand_dims(image, axis=0)

    #calculate and return keypoints
    keypoints_with_scores = run_movenet_inference(model_movenet, image)
    return JSONResponse(content = {'keypoints':keypoints_with_scores.tolist().__repr__()})


@app.get("/")
async def root():
    return dict(greetings='Hello')
