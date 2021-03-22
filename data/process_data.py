import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
     ETL (EXTACT): load data from two files and returned a merged veriosn 

     INPUT:
     messages_filepath: string for the messages dataset file path
     categories_filepath: string for the categories dataset file path

     OUTPUT:
     merged data frame 

    """

    # reading the messages information
    messages = pd.read_csv(messages_filepath)

    # reading the categories information
    categories = pd.read_csv(categories_filepath)

    # merging the two datasets together 
    df = messages.merge(categories, on="id")
    
    return df



def clean_data(df):
    """
    ETL (TRANSFORM): change the data by updating categorical colums and droping duplicates

    INPUT:
    df: pandas dataframe to work with

    OUTPUT:
    Transformed dataframe 
    """

    # creating a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=';', expand=True)

    # selecting the first row of the categories dataframe and using it
    # to extract and  rename categories columns. since all values end with - then a number
    # the - is used to get the first part of the string as column name
    row = categories.loc[0]
    category_colnames = row.apply(lambda x: x.split("-")[0])
    categories.columns = category_colnames

    # converting category values to 0s or 1s using negative indexing to 
    # get last character of a string
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = categories[column].astype("int")

    # replace values of (2) in related column with zero
    categories.loc[categories['related'] == 2, 'related'] = 0
    
    # replacing categories column in df with new columns (drop then Concatenate)
    df.drop(['categories'], axis=1, inplace = True)
    df = pd.concat([df, categories], axis = 1)

    # removing duplicates
    df.drop_duplicates(inplace = True)

    return df


def save_data(df, database_filename):
    """
    ETL (LOAD) : load the dataframe to an SQLite database using SQLAlchemy engine

    INPUT:
    df: extracted and transformed pandas dataframe
    database_filename: string file name for the database
    """
    
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('fig8data', engine, index=False, if_exists = 'replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()