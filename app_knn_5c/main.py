import os
import sys
import logging
import pandas as pd

from models.models import data_market
from fastapi import FastAPI
from predictor.predict_data import LoadAndPredict
from train.train_data import Clustering
from load.load_data import WebDataRetriever
from preprocess.preprocess_data import DataPreprocessor
from utilities.logging import MyLogger

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

logger = MyLogger("API Logs", logging.DEBUG)
app = FastAPI()

@app.get('/', status_code=200)
async def healthcheck():
    logger.info("5c Cluster Classifiers is ready!")
    return '5c Cluster Classifiers is ready!'

@app.post("/predict_5c/")
def predict_clusters(item: data_market):
    try: 
        # Logging prediction request
        logger.info("Received prediction request.")
        data = pd.DataFrame([[
            item.Education, 
            item.Income, 
            item.Kidhome, 
            item.Teenhome,
            item.Recency,
            item.MntWines, 
            item.MntFruits, 
            item.MntMeatProducts, 
            item.MntFishProducts,
            item.MntSweetProducts, 
            item.MntGoldProds, 
            item.NumDealsPurchases,
            item.NumWebPurchases, 
            item.NumCatalogPurchases, 
            item.NumStorePurchases,
            item.NumWebVisitsMonth, 
            item.AcceptedCmp3,
            item.AcceptedCmp4,
            item.AcceptedCmp5, 
            item.AcceptedCmp1, 
            item.AcceptedCmp2,
            item.Complain, 
            item.Response, 
            item.Age,
            item.Years_Since_Registration,
            item.Sum_Mnt, 
            item.Num_Accepted_Cmp, 
            item.Num_Total_Purchases
        ]])

        n_clusters = 5 

        predicted_cluster = LoadAndPredict.predict_clusters(data = data, n_clusters = n_clusters, models_dir='/app_knn/models/')
        
        # Logging prediction result
        logger.info(f"Predicted cluster: {int(predicted_cluster[0])}")
        return {"predicted_cluster": int(predicted_cluster[0])}
    
    except Exception as e:
        # Logging errors
        logger.error(f"An error occurred: {str(e)}")
        return {"error": "An error occurred during prediction."}

    
@app.get("/train_model_5c", status_code=200)
def train_model():
    try:
        # Logging training start
        logger.info("Training model started.")

        URL = 'https://raw.githubusercontent.com/juanchavezs/mlops_jpcs_proyectofinal/master/marketing_campaign.csv'
        DELIMITER = '\t'
        data_retriever = WebDataRetriever(url= URL, delimiter_url= DELIMITER , data_path= '/app_knn_5c/models/')
        data = data_retriever.retrieve_data()

        data_transform = DataPreprocessor.feature_generation(self='',data=data)

        clustering = Clustering()
        clustering.make_models(data_transform)
        
        # Logging training completion
        logger.info("Training model completed.")
        return "Trained model ready to go!"
    
    except Exception as e:
        # Logging errors during training
        logger.error(f"An error occurred: {str(e)}")
        return {"error": "An error occurred during training."}