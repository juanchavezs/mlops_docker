import pandas as pd
import os
import logging



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

    DATASETS_DIR = './data/'  # Directory where data will be saved.
    RETRIEVED_DATA = 'retrieved_data.csv'  # File name for the retrieved data.

    def __init__(self, url, data_path, delimiter_url):
        self.url = url
        self.DATASETS_DIR = data_path
        self.delimiter_url = delimiter_url


    def retrieve_data(self):
        # Configure logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s:%(name)s:%(module)s:%(levelname)s:%(message)s')
        file_handler = logging.FileHandler('load_data.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler) 
 
        try:
            # Logging data retrieval attempt
            logger.info("Retrieving data from URL...")

            # Loading data from specific URL
            data = pd.read_csv(self.url, delimiter=self.delimiter_url)

            # Create directory if it does not exist
            if not os.path.exists(self.DATASETS_DIR):
                os.makedirs(self.DATASETS_DIR)
                logger.info(f"Directory '{self.DATASETS_DIR}' created successfully.")
            else:
                logger.info(f"Directory '{self.DATASETS_DIR}' already exists.")

            # Save data to CSV file
            data.to_csv(self.DATASETS_DIR + self.RETRIEVED_DATA, index=False)

            # Logging data storage location
            logger.info(f'Data stored in {self.DATASETS_DIR + self.RETRIEVED_DATA}')
            
            print(f'Data stored in {self.DATASETS_DIR + self.RETRIEVED_DATA}')
     
        except Exception as e:
            # Logging errors during data retrieval and storage
            logger.error(f"An error occurred during data retrieval and storage: {str(e)}")

    
# Example
URL = 'https://raw.githubusercontent.com/juanchavezs/mlops_jpcs_proyectofinal/master/marketing_campaign.csv'
DELIMITER = '\t'
data_retriever = WebDataRetriever(url=URL, data_path='../data/', delimiter_url=DELIMITER)
data_retriever.retrieve_data()