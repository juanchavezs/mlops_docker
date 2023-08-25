import logging
from load.load_data import WebDataRetriever
from preprocess.preprocess_data import DataPreprocessor
from train.train_data import Clustering
from sklearn.pipeline import Pipeline
import pandas as pd
import os

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s:%(name)s:%(module)s:%(levelname)s:%(message)s')
file_handler = logging.FileHandler('main_proyect.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler) 

# Directories and URLs
DATASETS_DIR = './itesm_mlops/data/'
URL = 'https://raw.githubusercontent.com/juanchavezs/mlops_jpcs_proyectofinal/master/marketing_campaign.csv'
RETRIEVED_DATA = 'retrieved_data.csv'
DELIMITER = '\t'
TRAINED_MODEL_DIR = './itesm_mlops/models/'

if __name__ == "__main__":
    try:
        # Retrieving data
        logger.info("Retrieving data from URL...")
        data_retriever = WebDataRetriever(URL, DATASETS_DIR, DELIMITER)
        result = data_retriever.retrieve_data()
        logger.info("Data retrieval successful.")

        # Reading retrieved data
        df = pd.read_csv(DATASETS_DIR + RETRIEVED_DATA)

        # Data preprocessing
        logger.debug("Starting data preprocessing...")
        data_transformer = DataPreprocessor()
        pipeline = Pipeline([
            ('data_transformer', data_transformer),
        ])
        df_transform = pipeline.fit_transform(df)
        logger.debug("Data preprocessing completed.")

        # Model training
        logger.info("Starting model training...")
        clustering = Clustering()
        clustering.make_models(df_transform)
        logger.info("Model training completed.")

        logger.info("Process completed successfully.")
    except Exception as e:
        # Logging errors during the process
        logger.error(f"An error occurred: {str(e)}", exc_info=True)  # Also logging traceback
