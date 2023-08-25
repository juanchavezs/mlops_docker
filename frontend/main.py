import logging
import os
import sys

import requests
from fastapi import Body, FastAPI
from utilities.logging import MyLogger

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)

# adding the parent directory to
# the sys.path.
sys.path.append(parent)


logger = MyLogger("Main", logging.DEBUG, __name__)

app = FastAPI()


# ML model prediction function using the prediction API request
def predict_cluster(input):
    url3 = "http://app.docker:8000/predict"

    response = requests.post(url3, json=input)
    response = response.text

    return response


@app.get("/", status_code=200)
async def healthcheck():
    logger.info("ACTION->Front End Fraud Classifier is all ready to go!")
    return "Front End  Fraud Classifier is all ready to go!"


@app.post("/predict")
def predict(payload: dict = Body(...)):
    logger.debug(f"Incoming input in the front end: {payload}")
    response = predict_cluster(payload)
    logger.debug(f"Prediction: {response}")
    return {"response": response}

@app.get("/healthcheck")
async def v1_healhcheck():
    url3 = "http://app.docker:8000/"

    response = requests.request("GET", url3)
    response = response.text
    logger.info(f"Checking health: {response}")

    return response


@app.get("/train_model", status_code=200)
def train_model():
    url3 = "http://app.docker:8000/train_model"
    logger.info("Training model proccess...START")
    response = requests.request("GET", url3)
    response = response.text
    logger.info(f"Training model process ENDED. Result: {response}")

    return response

@app.get("/train_model_5c", status_code=200)
def train_model():
    url3 = "http://app.docker:8001/train_model_5c"
    logger.info("Training model proccess...START")
    response = requests.request("GET", url3)
    response = response.text
    logger.info(f"Training model process ENDED. Result: {response}")

    return response


@app.post("/predict")
def predictKNN(payload: dict = Body(...)):
    logger.debug(f"Incoming input in the front end: {payload}")
    response = predict_cluster_KNN(payload)
    logger.debug(f"Prediction: {response}")
    return {"response": response}


@app.get("/train_modelKNN", status_code=200)
def train_modelKNN():
    url3 = "http://appKNN.docker:8001/train_modelKNN"
    logger.info("Training model proccess...START")
    response = requests.request("GET", url3)
    response = response.text
    logger.info(f"Training model process ENDED. Result: {response}")

    return response