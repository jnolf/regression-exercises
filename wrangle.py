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
    ''' 
    This function acquires all necessary housing data from zillow 
    needed to better understand future pricing
    '''
    
    if os.path.exists('zillow.csv') and use_cache:
        print('Using cached csv')
        return pd.read_csv('zillow.csv')
    print('Acquiring data from SQL database')

    database_url_base = f'mysql+pymysql://{username}:{password}@{host}/zillow'
    query = '''
    SELECT bedroomcnt AS bedrooms, 
           bathroomcnt AS bathrooms, 
           calculatedfinishedsquarefeet AS sqft, 
           taxvaluedollarcnt AS tax_value, 
           yearbuilt AS yr_built,
           taxamount AS tax_amount,
           regionidcounty AS county_id,
           fips
        FROM properties_2017
    
        JOIN propertylandusetype USING(propertylandusetypeid)
        
        JOIN predictions_2017 pr USING (parcelid)
        WHERE propertylandusedesc IN ('Single Family Residential',
        
                                      'Inferred Single Family Residential')
                              AND pr.transactiondate LIKE '2017%%';
            '''
    
    
    df = pd.read_sql(query, database_url_base)
    df.to_csv('zillow.csv', index=False)
   
    return df

#################### Outliers (Hope This Works) ####################

def remove_outliers(df, k, col_list):
    ''' 
    This function remove outliers from a list of columns in a dataframe 
    and returns that dataframe
    '''
    
    # loop through each column
    for col in col_list:
        
        # Get the quantiles
        q1, q3 = df[col].quantile([.25, .75])
        
        # Get the quantile range
        iqr = q3 - q1
        
        # Establish the upper and lower
        upper_bound = q3 + k * iqr  
        lower_bound = q1 - k * iqr   

        # Redefine the DataFrame with removed outliers
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

############################# Clean ################################

def clean_zillow(df):
    '''
    This function takes in the zillow data, cleans it, and returns a dataframe
    '''
    
    # Apply a function to remove outliers
    df = remove_outliers(df, 1.5, ['bedrooms','bathrooms',
                                   'sqft','taxvalue','taxamount'])
    
    # Remove more of the outliers for sqft
    df = df[(df.sqft > 500) & (df.sqft < 2500)]
    # Remove more of the outliers for taxvalue
    df = df[(df.taxvalue > 500) & (df.taxvalue < 800000)]
    
    # Drop rows with null values since it is only a small portion of the
    dataframe 
    df = df.dropna()

    # Create list of datatypes I want to change
    int_cols = ['bedrooms','sqft','tax_amount','age']
    obj_cols = ['yr_built']
    
    # Change data types of above columns
    for col in df:
        if col in int_cols:
            df[col] = df[col].astype(int)
        if col in obj_cols:
            df[col] = df[col].astype(int).astype(object)
    
    # Drop the target column
    df = df.drop(columns='tax_value')
    
    return df 

############################# Split ################################

def split_data(df):
    '''
    
    '''
    
    train_val, test = train_test_split(df, train_size=0.8,random_state=123)
    
    
    train, validate = train_test_split(train_val, train_size=0.7, random_state=123)
    
    
    return train, validate, test 

############################ Wrangle ###############################

def wrangle_zillow():
    '''
    This function combines the acquire, clean and split portions of
    this file in order to be imported and quickly have data ready 
    for exploration.
    '''
    
    train, validate, test = split_data(clean_zillow(acquire_zillow()))

    return train, validate, test




