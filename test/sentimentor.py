"""
This python script puts together all the transformations needed for receiving a text input (comment or review) and
returning the sentiment associated. For this to be done, we load the pkl files for pipelines built on train.py script
and handle some common exceptions that we can face on production

* Metadata can be find at: https://www.kaggle.com/olistbr/brazilian-ecommerce
* Reference notebook: ../notebooks/EDA_BrazilianECommerce.ipynb

--- SUMMARY ---

1. Project Variables
2. Reading Data
3. Prep Pipelines
    3.1 Initial Preparation
    3.2 Text Transformers
4. Modeling
    4.1 Model Training
    4.2 Evaluating Metrics
    4.3 Complete Solution
    4.4 Final Model Performance
    4.5 Saving pkl Files

---------------------------------------------------------------
Written by Thiago Panini - Latest version: September 25th 2020
---------------------------------------------------------------
"""

# Importing libraries
import os
from joblib import load
from pandas import DataFrame
import pandas as pd
from getpass import getuser
from datetime import datetime


"""
-----------------------------------
------ 1. PROJECT VARIABLES -------
-----------------------------------
"""

# Variables for path address
PIPE_PATH = '../pipelines'
MODEL_PATH = '../models'
LOG_PATH = '../log_results/'

# Variables for pkl files
E2E_PIPE = 'text_prep_pipeline.pkl'
MODEL = 'sentiment_clf_model.pkl'


# Building up the class Sentimentor
class Sentimentor():

    def __init__(self, data):
        self.data = data
        self.pipeline = load(os.path.join(PIPE_PATH, E2E_PIPE))
        self.model = load(os.path.join(MODEL_PATH, MODEL))

        # Change the status of train attribute from the pipeline's vectorizer
        #self.pipeline.named_steps['text_features'].train = False

    def prep_input(self):
        """
        Takes an inputs (string or list) and applies the prep pipeline
        :return: updating the self.data attribute from the class
        """

        # Verify if the type of input data
        if type(self.data) is str:
            self.data = [self.data]
        elif type(self.data) is DataFrame:
            self.data = list(self.data.iloc[:, 0].values)

        # Apply the pipeline to prepare the input data
        return self.pipeline.transform(self.data)

    def make_predictions(self, export_results=False, export_path=LOG_PATH):
        """
        Takes the data input, applies the pipeline and the classifier model to return the sentiment
        :return:
        """

        # Preparing the data and calling the classifier for making predictions
        text_list = self.prep_input()
        pred = self.model.predict(text_list)
        proba = self.model.predict_proba(text_list)[:, 1]

        # Analyzing the results and preparing the output
        class_sentiment = ['Positive' if c == 1 else 'Negative' for c in pred]
        class_proba = [p if c == 1 else 1 - p for c, p in zip(pred, proba)]

        # Building up a pandas DataFrame to delivery the results
        results = {
            'text_input': self.data,
            'prediction': pred,
            'class_sentiment': class_sentiment,
            'class_probability': class_proba
        }
        df_results = DataFrame(results)

        # Exporting results
        if export_results:
            now = datetime.now().strftime('%Y%m%d_%H%M%S')
            df_results.to_csv(f'{export_path}{getuser()}_prediction_{now}.csv', index=False, sep=';', encoding='UTF-16')

        return df_results


if __name__ == '__main__':
    # Instancing an object and executing predictions
    text_input = 'Adorei O PRODUT0, me atendeu perfeitamente e pretendo adquirir mais itens nessa loja'
    test_input = pd.read_csv('test_data.csv', sep=';')
    sentimentor = Sentimentor(data=test_input)

    # Calling the method for preparing the input whatever its type
    output = sentimentor.make_predictions(export_results=True)

