import pandas as pd
import os
import logging
import sys

current_dir = os.getcwd()
sys.path.append(current_dir+'/app_knn/')

from utilities.logging import MyLogger

logger = MyLogger("Load_Logs", logging.DEBUG, 'Load_Logs')


class WebDataRetriever:
    """
    A class for retrieving data from a given URL.

    Parameters:
        url (str): The URL from which the data will be loaded.

    Attributes:
        url (str): The URL from which the data will be loaded.

    Example usage:
    ```
    URL = 'https://www.openml.org/data/get_csv/16826755/phpMYEkMl'
    data_retriever = WebDataRetriever(URL, DATASETS_DIR)
    result = data_retriever.retrieve_data()
    print(result)
    ```
    """
    def retrieve_data(url, delimiter_url, DATASETS_DIR, RETRIEVED_DATA):

        try:
            # Logging data retrieval attempt
            logger.info("Retrieving data from URL...")

            # Loading data from specific URL
            data = pd.read_csv(url, delimiter=delimiter_url)

            # Create directory if it does not exist
            if not os.path.exists(DATASETS_DIR):
                os.makedirs(DATASETS_DIR)
                logger.info(f"Directory '{DATASETS_DIR}' created successfully.")
            else:
                logger.info(f"Directory '{DATASETS_DIR}' already exists.")

            # Save data to CSV file
            data.to_csv(DATASETS_DIR + RETRIEVED_DATA, index=False)

            # Logging data storage location
            logger.info(f'Data stored in {DATASETS_DIR + RETRIEVED_DATA}')
            
            print(f'Data stored in {DATASETS_DIR + RETRIEVED_DATA}')
     
        except Exception as e:
            # Logging errors during data retrieval and storage
            logger.error(f"An error occurred during data retrieval and storage: {str(e)}")

    
# Example
URL = 'https://raw.githubusercontent.com/juanchavezs/mlops_jpcs_proyectofinal/master/marketing_campaign.csv'
DELIMITER = '\t'
PATH = '/data/'

data_retriever = WebDataRetriever.retrieve_data(url=URL, data_path= PATH, delimiter_url=DELIMITER)