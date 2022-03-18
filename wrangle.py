import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os
from env import username, password, host
import env

import warnings
warnings.filterwarnings("ignore")


def wrangle_zillow(use_cache=True):
    ''' This function acquires the data needed from Zillow on
    housing.
    '''
    
    if os.path.exists('zillow.csv') and use_cache:
        print('Using cached csv')
        return pd.read_csv('zillow.csv')
    print('Acquiring data from SQL database')

    database_url_base = f'mysql+pymysql://{username}:{password}@{host}/zillow'
    query ='''
    SELECT bedroomcnt AS bedrooms, 
           bathroomcnt AS bathrooms, 
           calculatedfinishedsquarefeet AS finished_sqft, 
           taxvaluedollarcnt AS tax_value,
           yearbuilt AS yr_built,
           taxamount AS tax_amount,
           fips
    FROM properties_2017
    WHERE propertylandusetypeid = '261';
    '''
    df = pd.read_sql(query, database_url_base)
    df.to_csv('zillow.csv', index=False)
   
    return df


def clean_zillow(df):
    df = df.dropna(subset =['bedrooms', 'bathrooms', 'finished_sqft', 'yr_built', 'fips'])
    df['tax_amount'].fillna(df.tax_amount.median(), inplace=True)
    df['tax_value'].fillna(df.tax_value.median(), inplace=True)
    
    return df


def split_zillow(df):
    '''
    Takes in a df
    Returns train, validate, and test DataFrames
    '''
    # Create train_validate and test datasets
    train_validate, test = train_test_split(df, 
                                        test_size=.2, 
                                        random_state=123)
    # Create train and validate datsets
    train, validate = train_test_split(train_validate, 
                                   test_size=.3, 
                                   random_state=123)

    print(f'train shape ==== {train.shape}')
    print(f'validate shape = {validate.shape}')
    print(f'test shape ===== {test.shape}')
    return train, validate, test




