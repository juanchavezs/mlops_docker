from sklearn.base import BaseEstimator, TransformerMixin
import os
import pandas as pd
from datetime import datetime 
from sklearn.cluster import KMeans
import joblib
import logging
import os 

# Create a 'logs' directory if it doesn't exist
if not os.path.exists("/itesm_mlops/logs"):
    os.makedirs("/itesm_mlops/logs")

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s:%(name)s:%(module)s:%(levelname)s:%(message)s')
file_handler = logging.FileHandler('/itesm_mlops/logs/train_data.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler) 

class Clustering:
    """
    A class for performing KMeans clustering analysis and saving models.

    Attributes:
    min_n_clusters (int): Minimum number of clusters for analysis.
    max_n_clusters (int): Maximum number of clusters for analysis.
    kmeans_distortions (list): List to store model distortions for different hyperparameters (n_clusters).
    models_dir (str): Directory to save trained models.

    Methods:
    load_kmeans_model(self, n_clusters): Load a KMeans clustering model from a file.
    predict_clusters(self, data, n_clusters): Predict clusters using a loaded KMeans model.
    make_models(self, df_transform): Build KMeans clustering models and save them.
    save_kmeans_model(self, model, n_clusters): Save a trained KMeans model to a file.
    """
    min_n_clusters = 2
    max_n_clusters = 8
    kmeans_distortions = []  # Model distortions for different hyperparameters(n_clusters)
    models_dir = './itesm_mlops/models/'  # Directory to save models

    def __init__(self):
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
            logger.info(f"Directory '{self.models_dir}' created successfully.")
        else: 
            logger.info(f"Directory '{self.models_dir}' already exists.")            

    def fit(self, data):
        return self 

    def save_kmeans_model(self, model, n_clusters) -> None:
        """
        Save the trained KMeans clustering model to a file.

        Parameters:
        model: Trained KMeans model to be saved.
        n_clusters (int): Number of clusters in the model.

        Returns:
        None

        Description:
        This method saves the trained KMeans clustering model to a file in the specified models directory.

        - 'model': Trained KMeans model to be saved.
        - 'n_clusters': Number of clusters associated with the model.
        - 'model_filename': Generate a filename for the model based on the number of clusters.
        - Save the KMeans model using the joblib.dump function.
        - Print a confirmation message indicating the successful saving of the model.
        """
        try:
            model_filename = f"{self.models_dir}kmeans_model_{n_clusters}.pkl"
            joblib.dump(model, model_filename)
            logger.info(f"Saved KMeans model with {n_clusters} clusters to {model_filename}")
        except Exception as e:
            logger.error(f"An error occurred while saving KMeans model: {str(e)}")


          
    def make_models(self, df_transform: pd.DataFrame) -> None:
        """
        Build KMeans clustering models for various numbers of clusters and save the models.

        Parameters:
        df_transform (pd.DataFrame): DataFrame with transformed data for clustering.

        Returns:
        None

        Description:
        This method constructs KMeans clustering models with varying numbers of clusters
        and evaluates the models using the silhouette score. It also saves the KMeans models
        for future use.

        - 'df_transform': DataFrame containing the transformed data for clustering.
        - 'metrics': A list to store silhouette scores for different numbers of clusters.
        - Iterate through a range of cluster numbers from 'min_n_clusters' to 'max_n_clusters'.
        - Create a KMeans model with the specified number of clusters and fit it to the transformed data.
        - Calculate the silhouette score for the current model and data.
        - Save the inertia (distortion) of the KMeans model for later use.
        - Save the trained KMeans model using the 'save_kmeans_model' method.
        """
        try:
            logger.info("Building KMeans clustering models and saving them...")
            
            for n_clusters in range(Clustering.min_n_clusters, Clustering.max_n_clusters + 1):
                kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, max_iter=280, random_state=42)
                pred = kmeans.fit_predict(df_transform)
                Clustering.kmeans_distortions.append(kmeans.inertia_)
                self.save_kmeans_model(kmeans, n_clusters)  # Save KMeans model

            logger.info("KMeans clustering models built and saved.")
        except Exception as e:
            logger.error(f"An error occurred during KMeans model building and saving: {str(e)}")

# # Usage
# if __name__ == "__main__":
#     """
#     Demonstration of using the Clustering class for prediction and preprocessing.

#     Description:
#     This part of the code showcases the usage of the Clustering class to predict clusters for new data
#     and save the preprocessed data along with the predicted clusters to a new CSV file.

#     - Create an instance of the Clustering class.
#     - Load new data for prediction from a CSV file.
#     - Perform data preprocessing steps using various functions, including data reduction and scaling.
#     - Specify the number of clusters to predict.
#     - Predict clusters for the new data using the 'predict_clusters' method.
#     - Add the predicted cluster labels to the original DataFrame.
#     - Save the preprocessed data with predicted clusters to a new CSV file.

# df_transform = DataPreprocessor.feature_generation(df)
# clustering = Clustering()
# clustering.make_models(df_transform)