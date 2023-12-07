from fastapi import FastAPI, Request, File, UploadFile
from typing import List, Dict
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from tensorflow import convert_to_tensor
from tensorflow.keras.models import load_model as tfk__load_model

from ptai.ml_logic.model import prediction_model
from ptai.movenet.movenet_model import load_model_from_tfhub, run_movenet_inference
from ptai.ml_logic.posedetect import load_posedetect_model
from ptai.movenet.movenet_main import angle_calc
from PIL import Image

import tensorflow as tf
import io
import numpy as np
import pandas as pd
import json


# Create the app so fastapi knows what to execute
app = FastAPI()

#no clue
origins = ['*']

# Create the class that defines hoe the classifier receives item

#no clue
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Add the CNN model
#"""
#model_CNN = tfk__load_model('/ptai-proj/raw_data/models/model_1.h5', compile=False)
#model_CNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#app.state.model = model_CNN
#"""

# Adding different classifier model to run tests
model_class = load_posedetect_model('/ptai-proj/raw_data/models/best_automl_model.pkl')
app.state.model = model_class
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
    keypoint_angles = angle_calc(keypoints_with_scores)
    return JSONResponse(content = {'keypoints':pd.DataFrame(keypoint_angles, index=[0]).to_json(),
                                   "keypoints_scores":json.dumps(keypoints_with_scores.__repr__())})

class DictionaryModel(BaseModel):
    data: Dict[str, str]

# Creating a new enpoint for the new model
@app.post("/automl_model/")
async def predicting_the_move(dict_data: DictionaryModel):
    # initiate model
    model = app.state.model
    assert model is not None
    # start the prediction

    data_api = dict_data.data
    keypoint_dict = eval(data_api["keypoints"])
    df=pd.DataFrame(keypoint_dict, index=[0])
    y_pred = model.predict(df)
    return {"predict":f"{y_pred}"}




@app.get("/")
async def root():
    return dict(greetings='Hello')
