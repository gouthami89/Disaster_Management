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

from sklearn.preprocessing import MultiLabelBinarizer as MLB
from sklearn.multiclass import OneVsRestClassifier as ORC

from sklearn.metrics import accuracy_score

import pickle

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('database', con=engine)
    X = df['message'].values
    # target is a set of category names
    category_names = df.columns[4:]
    Y = df[category_names].values
    return X, Y, category_names

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', ORC(RandomForestClassifier(n_estimators=10)))
                        ])
    return pipeline

def evaluate_model(model, X_test, Y_test):
    Y_pred = model.predict(X_test)
    return accuracy_score(Y_test, Y_pred)

def save_models(models, model_filepath):
    # save the classifiers for each class
    with open(model_filepath, 'wb') as f:
        for model in models:
             pickle.dump(model, f)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building and Training models...')
        models = []
        for c, category in enumerate(category_names):
            model = build_model()
            model.fit(X_train, Y_train[:, c])
            models.append(model)
            test_accuracy = evaluate_model(model, X_test, Y_test[:, c])
            print('Accuracy score for the category `{}` is {}'.format(category, round(test_accuracy, 3)))

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_models(models, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
