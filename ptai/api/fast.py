from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from tensorflow import convert_to_tensor
from ptai.ml_logic.model import load_model, prediction_model
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model as tfk__load_model
import tensorflow as tf
import io
import numpy as np
from PIL import Image
import pandas as pd
from typing import Annotated
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


model = tfk__load_model('/home/kyrill/code/pt-ai/pt-ai/raw_data/models/model.h5', compile=False)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
app.state.model = model

@app.post("/uploadfile/")
async def prediction(file: UploadFile = File(...)):

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    img_array = convert_to_tensor(np.array(image), dtype=tf.float32)

    #message_shape = img_array.shape

    #image = Image.open(io.BytesIO(file)).convert("RGB")
    model = app.state.model
    assert model is not None
    y_pred = prediction_model(model, img_array)
    return JSONResponse(content = {'pose':f'Predicted pose is: {y_pred}'})

    #return JSONResponse(content = {"message":f"All g, image shape is {message_shape}"})


@app.get("/")
async def root():
    return dict(greetings='Hello')
