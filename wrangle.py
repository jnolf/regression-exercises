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

############################# Acquire ###############################

def acquire_zillow(use_cache=True):
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

############################# Clean ################################

def clean_zillow(df):
    '''
    This function takes in the zillow data, cleans it, and returns a dataframe
    '''
    
    # Rename some columns for simplicity
    df = df.rename(columns={'bedroomcnt':'bedrooms', 'bathroomcnt':'bathrooms', 
                            'calculatedfinishedsquarefeet':'area',
                            'taxvaluedollarcnt':'taxvalue'})
    # Apply a function to remove outliers
    df = remove_outliers(df, 1.5, ['bedrooms', 'bathrooms','area','taxvalue','taxamount'])
    
    # Remove more of the outliers for area
    df = df[(df.area > 500) & (df.area < 2500)]
    # Remove more of the outliers for taxvalue
    df = df[(df.taxvalue > 500) & (df.taxvalue < 800000)]
    
    # Drop rows with null values since it is only a small portion of the dataframe 
    df = df.dropna()

    # create age column based on yearbuilt
    df['age'] = 2021 - df.yearbuilt
    
    # Create list of datatypes I want to change
    int_col_list = ['bedrooms','area','taxvalue','age']
    obj_col_list = ['yearbuilt','fips']
    
    # Change data types where it makes sense
    for col in df:
        if col in int_col_list:
            df[col] = df[col].astype(int)
        if col in obj_col_list:
            df[col] = df[col].astype(int).astype(object)
    
    # drop taxamount since we will be predicting tax value and tax amount is considered data leakage
    df = df.drop(columns='taxamount')

    # Encode FIPS column and concatenate onto original dataframe
    dummy_df = pd.get_dummies(df['fips'], drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    
    return 

############################# Split ################################

def split_data(df, random_state=123, stratify=None):
    '''
    This function takes in a dataframe and splits the data into train, validate and test samples. 
    Test, validate, and train are 20%, 24%, & 56% of the original dataset, respectively. 
    '''
   
    if stratify == None:
        # split dataframe 80/20
        train_validate, test = train_test_split(df, test_size=.2, random_state=random_state)

        # split larger dataframe from previous split 70/30
        train, validate = train_test_split(train_validate, test_size=.3, random_state=random_state)
    else:

        # split dataframe 80/20
        train_validate, test = train_test_split(df, test_size=.2, random_state=random_state, stratify=df[stratify])

        # split larger dataframe from previous split 70/30
        train, validate = train_test_split(train_validate, test_size=.3, 
                            random_state=random_state,stratify=train_validate[stratify])

    # results in 3 dataframes
    return train, validate, test 

############################ Wrangle ###############################

def wrangle_zillow():
    '''
    This function acquires, clean, and splits the zillow data and returns it ready for exploration
    '''
    
    train, validate, test = split_data(clean_zillow(acquire_zillow()))

    return train, validate, test




