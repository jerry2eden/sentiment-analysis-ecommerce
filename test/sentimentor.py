"""
Class creation for dealing with production text inputs and return its sentiment (positive or negative) and also
the prediction probability
"""

# Importing libraries
from joblib import load
from pandas import DataFrame
import pandas as pd
from getpass import getuser
from datetime import datetime


# Building up the class Sentimentor
class Sentimentor():

    def __init__(self, data, pipeline_path='../pipeline/text_prep_pipeline.pkl',
                 clf_path='../model/sentiment_classifier.pkl'):
        self.data = data
        self.pipeline = load(pipeline_path)
        self.clf = load(clf_path)

        # Change the status of train attribute from the pipeline's vectorizer
        self.pipeline.named_steps['text_features'].train = False

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
        return self.pipeline.fit_transform(self.data)

    def make_predictions(self, export_results=False, export_path='../log_results/'):
        """
        Takes the data input, applies the pipeline and the classifier model to return the sentiment
        :return:
        """

        # Preparing the data and calling the classifier for making predictions
        text_matrix = self.prep_input()
        pred = self.clf.predict(text_matrix)
        proba = self.clf.predict_proba(text_matrix)[:, 1]

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
    text_input = 'Não gostei do produto e achei caro. PAguei R$99,00 reais por algo de baixa qualidade'
    text_input = pd.read_csv('../data/train_data.csv', sep=';', usecols=['review_comment_message'])
    sentimentor = Sentimentor(data=text_input)

    # Calling the method for preparing the input whatever its type
    output = sentimentor.make_predictions(export_results=True)

