import pandas as pd
from sklearn.model_selection import  train_test_split

data=pd.read_csv('processed_data/feature_engineered_data/feature_engineered_data.csv')

def train_test_split(data):
    cols_to_drop=['Revenue']
    for i in cols_to_drop:
        data.drop(columns=[i],inplace=True)
    x=data.drop('Units_Sold')
    y=data['Units_Sold']