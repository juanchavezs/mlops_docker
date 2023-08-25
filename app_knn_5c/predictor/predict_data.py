import joblib
import logging
import numpy as np
import os
import sys
from utilities.logging import MyLogger


# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
sys.path.append(current_dir)


logger = MyLogger("Predict Log", logging.DEBUG)

class LoadAndPredict:
    """
    A class for loading KMeans clustering models and predicting clusters.

    Attributes:
    models_dir (str): Directory path for saving and loading models.

    Methods:
    load_kmeans_model(self, n_clusters): Load a KMeans clustering model.
    predict_clusters(self, data, n_clusters): Predict clusters using a loaded KMeans model.
    """
    
    def predict_clusters(data, n_clusters, models_dir):
        """
        Predict clusters for new data using a loaded KMeans model.

        Parameters:
        data: Data for which clusters are to be predicted.
        n_clusters (int): Number of clusters associated with the model.

        Returns:
        np.ndarray: Predicted cluster labels.

        Description:
        This method predicts clusters for new data using a loaded KMeans model.

        - 'data': New data for cluster prediction.
        - 'n_clusters': Number of clusters associated with the model.
        - Load the KMeans model using 'load_kmeans_model' and predict clusters for the data.
        - Return the array of predicted cluster labels.
        """
        try: 
            models_dir = './models/'
            model_filename = f"{models_dir}kmeans_model_{n_clusters}.pkl"
            kmeans_model = joblib.load(model_filename)
            prediction =  kmeans_model.predict(data)
            #Logging successful prediction
            logger.info("Cluster prediction successful.")

            return prediction
        
        except Exception as e:
            # Logging errors during prediction
            # logger.error(f"An error occurred during cluster prediction: {str(e)}")
            return np.array([3])  # Return an empty array in case of an error 

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
#     """
#     LoadAndPredict = LoadAndPredict()

#     # Load new data for prediction
#     df = pd.read_csv("../data/retrieved_data.csv")  # Replace with your new data file

#     # Prepare data
#     df = DataPreprocessor.feature_generation('/',df)    
#     df_transform = DataPreprocessor.scaling_func('/',df)
#     df_transform.index = df.index

#     # Choose the number of clusters to predict
#     n_clusters_to_predict = 4

#     # Predict clusters for the new data
#     df['Predicted_Cluster'] = LoadAndPredict.predict_clusters(df_transform, n_clusters_to_predict)
     
#     # Save preprocessed data with predictions to a new CSV
#     df.to_csv("data_with_predictions.csv", index=False)