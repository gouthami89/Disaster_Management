import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages.merge(categories, on='id')

def clean_data(df):
    # generate separate category columns
    separated_categories = pd.DataFrame(df['categories'].str.split(';', expand=True))
    separated_category_names = [name[0] for name in separated_categories.loc[0].str.split('-')]
    separated_categories.columns = separated_category_names

    # drop label from entry in each category column
    for col in separated_categories:
        separated_categories[col] = [int(c[1]) for c in separated_categories[col].str.split('-')]

    # drop categories from original df
    df = df.drop(['categories'], axis=1)

    # Merge df with separated_categories
    for column in separated_categories:
        df[column] = separated_categories[column]

    # Drop duplicates
    df.drop_duplicates(inplace=True)
    #
    # # Drop rows with NaN values
    # df.dropna(inplace=True)

    return df

def save_data(df, database_filename):
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('database', engine, index=False)


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
