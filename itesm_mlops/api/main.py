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

    predicted_cluster = LoadAndPredict.predict_clusters(data = data, n_clusters = n_clusters)
    return {"predicted_cluster": int(predicted_cluster[0])}
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)