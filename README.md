# Disaster_Management
NLP on news, twits for disaster management

### Disaster Response Pipeline Project
In this project, we use the NLTK package to tokenize and vectorize text messages (news, twits).
The resulting tfidf matrices are then trained against the targets 
(multiclass, multioutput labels corresponding to the disaster aid category).

The goal is to eventually arrive at a model that will appropriately classify an incoming
text message into as many disaster aid categories as relevant to the text.

The module consists of 3 segments:
1) Data processing - where messages and their corresponding categories are saved as pandas dataframes
2) Modeling pipeline - it is here where all the work is done. The steps here are:
	  a) Tokenizing text
    b) applying countvectorizers that uses the tokenizer to convert text documents into a matrix of token counts
    c) converting above count matrix into normalized tf-idf matrices
    d) Applying a machine learning model to the training dataset, while grid searching for suitable hyper parameter
    e) Evaluating the model on testing dataset
    f) Saving the model
3) Using the trained model pipeline on a Flask app for user friendly interface

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
