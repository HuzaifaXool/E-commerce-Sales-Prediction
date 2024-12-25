import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler



le=LabelEncoder()
Oe=OneHotEncoder(sparse_output=False)
Ss=StandardScaler()

df="processed_data/cleaned_data.csv"

data=pd.read_csv(df)

def remove_cols(data):
    data=data.drop(columns=['Month','Day','year'])
    print(data.head())
    return data


def ordinal_data(data):
    customer_order = ['Occasional', 'Regular', 'Premium']
    order = {customer_order[i]: i for i in range(len(customer_order))}
    data['Customer_Segment'] = data['Customer_Segment'].map(order)
    return data


def non_ordinal_data(data):
    encoded_data=Oe.fit_transform(data[['Product_Category']]).astype(int)
    encoded_df=pd.DataFrame(encoded_data,columns=Oe.get_feature_names_out(['Product_Category']))
    data=pd.concat([data,encoded_df],axis=1)
    data=data.drop(columns=['Product_Category'],axis=1)
    return data

def scalling_cols(data):
    cols=['Price','Marketing_Spend','Discount']
    for i in cols:
        data[i]=Ss.fit_transform(data[[i]])
    return data

data=remove_cols(data)
data=ordinal_data(data)
data=non_ordinal_data(data)
data=scalling_cols(data)

dir_path = 'processed_data/feature_engineered_data'

if not os.path.exists(dir_path):
    os.makedirs(dir_path)


file_name = "feature_engineered_data.csv"
file_path = os.path.join(dir_path, file_name)

data.to_csv(file_path, index=False, encoding='utf-8')

print(f"File saved at: {file_path}")




