import pandas as pd
import numpy as np
import os


def load_data(file_path:str):
    data=pd.read_csv(file_path)
    print("Data read to Dataframe\n")
    return data

def data_summary(data):
    print("\nData Summary of the dataset is as follows\n")
    print("="*60)
    print(f"The data set consists of\n\nrows-->{data.shape[0]} columns-->{data.shape[1]}\n\n")
    print("\nDataTypes are:\n")
    print(data.dtypes)
    print("\nData statistics are:\n")
    print(data.describe())
    print("\nMissing values counts:\n")
    print(data.isna().sum())
    print("\n")
    print("="*60)
    print("\n")
    return data


def missing_values(data):
    for i in data.columns:
        if np.issubdtype(data[i].dtype, np.number):  
            if data[i].isnull().sum() > 1:
                msg = (f"The column {i} consists of Numerical data and has {data[i].isnull().sum()} null values. Replacing with the mean value of {i}: {data[i].mean()}")
                print(msg)
                data[i].fillna(data[i].mean(), inplace=True)
            else:
                msg = f"No null values found in {i}. All good!"
                print(msg)
        else:
            if data[i].isnull().sum() > 1:
                msg = (f"The column {i} consists of Categorical data and has {data[i].isnull().sum()} null values. Replacing with the mode value of {i}: {data[i].mode()[0]}")
                print(msg)
                data[i].fillna(data[i].mode()[0], inplace=True)
            else:
                msg = f"No null values found in {i}. All good!"
                print(msg)
                
    return data

def del_duplicates(data):
    if data.duplicated().sum()>0:
        print(f'\nThis data set contains {data.duplicated().sum()} duplicate values dropping them\n')
        data=data.drop_duplicates()
    else:
        print(f"\nNo Duplicate values found....!\n")
    return data
        



def handling_outliers(data):
    numerical_cols=[i for i in data.select_dtypes('number')]
    for i in numerical_cols:
        q1=data[i].quantile(0.25)
        q3=data[i].quantile(0.75)
        iqr=q3-q1
        lower_bond=q1-1.5*iqr
        upper_bond=q3+1.5*iqr
        data[i]=data[i].clip(lower=lower_bond,upper=upper_bond)
    return data

def date_cols(data):
    data['Date']=pd.to_datetime(data['Date'],format="%d-%m-%Y")
    data['Month']=data['Date'].dt.month
    data['year']=data['Date'].dt.year
    data['Day']=data['Date'].dt.day
    return data

def per_unit_price(data):
    data['unit_price']=round(data['Price']/data['Units_Sold'])
    print(f"Modified data set consists of\n\nrows-->{data.shape[0]} columns-->{data.shape[1]}\n\n")
    return data


output_folder="processed_data"
file_name="cleaned_data.csv"
output_path=os.path.join(output_folder,file_name)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

file_path='data/Ecommerce_Sales_Prediction_Dataset.csv'
data=load_data(file_path)
data=data_summary(data)
data=missing_values(data)
data=del_duplicates(data)
data=handling_outliers(data)
data=date_cols(data)
data=per_unit_price(data)

data.to_csv(output_path,index=False)

print(f"\nProcessed data saved to {output_path}\n")