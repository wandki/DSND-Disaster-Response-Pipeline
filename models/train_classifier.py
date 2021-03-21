import sys

from sqlalchemy import create_engine
import pandas as pd
import numpy as np

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import pickle


def load_data(database_filepath):
    """
    load the dataframe from the SQLite database using SQLAlchemy engine
    """
    engine = create_engine('sqlite:///'+ database_filepath)
    df = df = pd.read_sql_table("fig8data", engine)
    
    X = df.message
    Y = df.iloc[:,4:]
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
    """
    function reference: this is a funcion reused from Udacity data science nanodegree
    to apply tokenize, lemmatize, case chanch and stripping
    
    INPUT: 
    text: text to tokenize
    
    OUTPUT:
    clean_tokens: new text tp be used
    """

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    create the pipline and find best parameters
    
    OUTPUT: model with best parameters 
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters =  {'clf__estimator__n_estimators': [100, 150],
                    'clf__estimator__min_samples_split': [2, 4]
                  }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv
    


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluating the model
    
    INPUT:
    model: classification model
    X_test: test data (messages)
    Y_test: expected classification 
    category_names: list of categories
    """
    Y_pred = model.predict(X_test)
    
    for i, col in enumerate(category_names):
        print(classification_report(Y_test[col], Y_pred.T[i]))
    


def save_model(model, model_filepath):
    """
    save model to pickle file
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()