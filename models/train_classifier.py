'''
This is a collection of functions that train on text messages
to predict aid category for each reported text message
'''
import sys

import pandas as pd
import numpy as np

from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import MultiLabelBinarizer as MLB
from sklearn.multiclass import OneVsRestClassifier as ORC
from sklearn.multioutput import MultiOutputClassifier as MOC

from sklearn.grid_search import GridSearchCV as GS
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

import pickle

def load_data(database_filepath):
    '''
    Load the saved SQL database
    Args:
        database_filepath: database filepath
    Returns:
        X: tokenized text (features)
        Y: disaster aid categories (multiclass multioutput targets)
        category_names: list of disaster aid categories 
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('database', con=engine)
    X = df['message'].values
    # target is a set of category names
    category_names = df.columns[4:]
    Y = df[category_names].values
    return X, Y, category_names

def tokenize(text):
    '''
    Tokenize the text message and remove stop words, captialization etc.
    Args:
        text: input text message
    Returns:
        clean_tokens: tokenized list of strings
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model(ML_model=None):
    '''
    Build the training pipeline 
    Args:
    Returns:
        pipeline: pipeline consisting of count vectorizer, tfidf transformer and ML model
    '''
    if ML_model:
        if ML_model['type'] == 'RF':
            clf = MOC(RandomForestClassifier(**ML_model['params']))
        elif ML_model['type'] == 'SVC':
            clf = MOC(SVC(**ML_model['params']))
        else:
            raise ValueError('Invalid ML model specified')
    else:
        clf = MOC(RandomForestClassifier(n_estimators=10))
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', clf)
                        ])
    return pipeline

def custom_score(Y_true, Y_pred):
    category_scores = []
    for c in range(Y_true.shape[1]):
        category_scores.append(accuracy_score(Y_true[:, c], Y_pred[:, c]))
    return np.mean(category_scores)

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate the trained model (pipeline) on testing dataset
    Args:
        model: pipeline consisting of count vectorizer, tfidf transformer and ML model
        X_test: testing dataset features
        Y_test: testing dataset target values
        category_names: list of disaster aid categories
    Returns:
    '''
    Y_pred = model.predict(X_test)
    for c, cat in enumerate(category_names):
        print('Scores for the category "{}"\n'.format(cat))
        print('Accuracy is {}'.format(round(accuracy_score(Y_test[:,c], Y_pred[:,c]), 2)))
        if cat != 'related':
            print('Precision is {}'.format(round(precision_score(Y_test[:,c], Y_pred[:,c]), 2)))
            print('Recall is {}'.format(round(recall_score(Y_test[:,c], Y_pred[:,c]), 2)))
            print('F1 is {}'.format(round(f1_score(Y_test[:,c], Y_pred[:,c]), 2)))
        print('\n')
        
def save_model(model, model_filepath):
    '''
    Save model as a pickle file 
    Args:
        model: pipeline consisting of count vectorizer, tfidf transformer and ML model
        X_test: testing dataset features
        Y_test: testing dataset target values
        category_names: list of disaster aid categories
    Returns:
    '''

    # save the classifiers for each class
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

def main():
    '''
    Function executing steps required to train, evaluate and save prediction model
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        
        # Load data and split to train and test fractions
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('BUILDING AND TRAINING MODEL PIPELINE:')
        model = build_model(ML_model={'type': 'RF', 'params': {'n_estimators': 20}})

        print('Performing grid search for suitable ML model hyper parameters...')
        parameters = {'clf__estimator__n_estimators': [20, 40]}
        accuracy_sc = make_scorer(custom_score)
        model_gs = GS(model, parameters, scoring=accuracy_sc)
        model_gs.fit(X_train, Y_train)
        best_model = model_gs.best_estimator_

        print('Evaluating trained pipeline on testing dataset...')
        test_accuracy = evaluate_model(best_model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(best_model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

