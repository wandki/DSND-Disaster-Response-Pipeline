# Disaster Response Pipeline Project

### Project Overview
This project covers figure eight data analysis and new message classification using ETL, NLP and ML pipelines.  the results are presented in a web app that provides:
 1. new message classification
 2. figure based on the data


### Installation
Run all py files with Python3. Make sure that you instal the following libraries using pip3 install command:
* pandas
* sklearn
* nltk
* sqlalchemy
* pickle
* Flask
* plotly
* joblib
* sys
* json


### Project structure

data:
- DisasterResponse.db :  SQLite DataBase  which is the output of the ETL pipeline
- process_data.py : ETL pipeline
- css input files from figure 8

Models:
- classifier.pkl : pickle file which is the output of the ML pipeline
- train_classifier.py : ML pipeline

app:
run.py : Flask app 
 

### How to run
First run  the ETL pipeline from the data directory to clean data and store it in database: 
python3 process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

Then run the ML pipeline from the models directory to train classifier and save it:
python3 train_classifier.py ../data/DisasterResponse.db classifier.pkl

Finally run the web app from the app directory:
python3 run.py

and  open the url   http://0.0.0.0:3001/ in your web browser


### Acknowledgements
- Data provided by figure-8 
- Tokenize function and other parts & knowledge by Udacity  
