import os
import sys
from models.models import data_market
import pandas as pd
from fastapi import FastAPI

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from predictor.predict_data import LoadAndPredict
predictor =LoadAndPredict()
from train.train_data import Clustering
from load.load_data import WebDataRetriever
from preprocess.preprocess_data import DataPreprocessor

app = FastAPI()

@app.get('/', status_code=200)
async def healthcheck():
    return 'Cluster Classifiers is ready!'

@app.post("/predict/")
def predict_clusters(item: data_market):
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

    n_clusters = 4  

    predicted_cluster = LoadAndPredict.predict_clusters(data = data, n_clusters = n_clusters, models_dir='../models/')
    return {"predicted_cluster": int(predicted_cluster[0])}
    
@app.get("/train_model", status_code=200)
def train_model():
    try: 
        URL = 'https://raw.githubusercontent.com/juanchavezs/mlops_jpcs_proyectofinal/master/marketing_campaign.csv'
        DELIMITER = '\t'
        data_retriever = WebDataRetriever(url= URL, delimiter_url= DELIMITER , data_path= '../data/')
        data = data_retriever.retrieve_data()

        data_transformer = DataPreprocessor()
        data_transform = DataPreprocessor.feature_generation(data)

        clustering = Clustering()
        clustering.make_models(data_transform)
    except: 
        print('No')

    return "Trained model ready to go!"

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)