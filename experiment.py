'''
@author     Diego Jurado
@project    CS1699 / CS 1951 Term Research Project
@title      Compensating for Annotation Bias via Re-Annotation
'''

# Imports
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Main

# Data Processing
df = import_data("dataset.csv")

# Data Set Info
# print Total Length
# Proportion of Hate Speech, Proportion of Not Hate Speech

# Iterate through Trials




# Helper Methods

# Data Processing
# params: String file
# return: DataFrame df
def import_data(file):
    df = pd.read_csv(file).drop_duplicates
    # randomly sample 3000 hate speech
    hate = df.query('Class == "hate"').sample(n=3000).copy() 
    # randomly sample 3000 non-hate speech
    non_hate = df.query('Class == "none"').sample(n=3000).copy()
    # merge hate and non-hate
    df = pd.merge(hate, non_hate, how='outer').copy()

    print("\nAny values in experiment DataFrame null:", df.isnull.values.any())

    return df

def data_stat(df):
    length = len(df)
    h_length = len(df.query('Class == "hate"'))
    n_length = len(df.query('Class == "none"'))
    print("\nNumber of samples:", length,"\nNumber of Hate Samples:", h_length, "\nNumber of non-Hate Samples")

# Randomized Re-Annotation Sampling
# params: Dataframe df, float fn, float fp
# return:

# Intelligent Re-Annotation Sampling
# params: Dataframe df, float fn, float fp
# return:

# Randomized Quantity Selection
# return: 

# Intelligent Quantity Selection
# params: 
# return:
