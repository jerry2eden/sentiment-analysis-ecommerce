"""
Script for training a sentimental analysis model based on online e-commerce reviews.
This training scripts includes:

    1. Reading the training data available for this task
    2. Building and saving up a text prep pipeline
    3. Training and saving up a classification model
"""

# Importing libs
import pandas as pd
import numpy as np
from utils.text_utils import re_breakline, re_dates, re_hiperlinks, re_money, re_negation, re_numbers, \
    re_special_chars, re_whitespaces, ApplyRegex, StemmingProcess, \
    StopWordsRemoval, TextFeatureExtraction
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
from sklearn.linear_model import LogisticRegression
from utils.ml_utils import BinaryClassifiersAnalysis


# Reading the data with the text corpus and dropping the null comments
df = pd.read_csv('../../data/olist_order_reviews_dataset.csv', usecols=['review_score', 'review_comment_message'])
df.dropna(inplace=True)

# Labeling training data based on review score
score_map = {
    1: 'negative',
    2: 'negative',
    3: 'positive',
    4: 'positive',
    5: 'positive'
}
df['label'] = df['review_score'].map(score_map)
df['target'] = df['label'].apply(lambda x: 1 if x == 'positive' else 0)

# Building a prep pipeline
# Defining regex transformers to be applied
regex_transformers = {
    'break_line': re_breakline,
    'hiperlinks': re_hiperlinks,
    'dates': re_dates,
    'money': re_money,
    'numbers': re_numbers,
    'negation': re_negation,
    'special_chars': re_special_chars,
    'whitespaces': re_whitespaces
}

# Defining the vectorizer to extract features from text
pt_stopwords = stopwords.words('portuguese')
vectorizer = TfidfVectorizer(max_features=300, min_df=7, max_df=0.8, stop_words=pt_stopwords)

# Building the Pipeline
text_pipeline = Pipeline([
    ('regex', ApplyRegex(regex_transformers)),
    ('stopwords', StopWordsRemoval(pt_stopwords)),
    ('stemming', StemmingProcess(RSLPStemmer())),
    ('text_features', TextFeatureExtraction(vectorizer))
])

# Preparing the data and putting it into the Pipeline
X = list(df['review_comment_message'].values)
y = df['target'].values

# Applying and saving the text prep pipeline
X_processed = text_pipeline.fit_transform(X)
joblib.dump(text_pipeline, '../../pipeline/text_prep_pipeline.pkl')
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=.20, random_state=42)

# Training a Logistic Regression model for classifying the sentiment from comments
logreg_param_grid = {
    'C': np.linspace(0.1, 10, 20),
    'penalty': ['l1', 'l2'],
    'class_weight': ['balanced', None],
    'random_state': [42],
    'solver': ['liblinear']
}

# Setting up the classifiers
set_classifiers = {
    'LogisticRegression': {
        'model': LogisticRegression(),
        'params': logreg_param_grid
    }
}

# Creating an object and training the classifiers
clf_tool = BinaryClassifiersAnalysis()
clf_tool.fit(set_classifiers, X_train, y_train, random_search=True, scoring='accuracy')

# Evaluating metrics
df_performances = clf_tool.evaluate_performance(X_train, y_train, X_test, y_test, cv=5)
df_performances.reset_index(drop=True).style.background_gradient(cmap='Blues')
df_performances.to_csv('results/model_performance.csv', index=False)

# Saving the LogisticRegression model (already analyzed on jupyter notebook)
model = clf_tool.classifiers_info['LogisticRegression']['estimator']
joblib.dump(model, '../../model/sentiment_classifier.pkl')