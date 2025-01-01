import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

le=LabelEncoder()
Oe=OneHotEncoder(sparse_output=False)
Ss=StandardScaler()

df_path="processed_data/cleaned_data.csv"
data=pd.read_csv(df_path)

def ordinal_data(data):
    customer_order=['Occasional','Regular','Premium']
    order={customer_order[i]:i for i in range(len(customer_order))}
    data['Customer_Segment']=data['Customer_Segment'].map(order)
    return data

def non_ordinal_data(data):
    encoded_data=Oe.fit_transform(data[['Product_Category']]).astype(int)
    encoded_df=pd.DataFrame(encoded_data,columns=Oe.get_feature_names_out(['Product_Category']))
    data=pd.concat([data,encoded_df],axis=1)
    data=data.drop(columns=['Product_Category'],axis=1)
    return data

def scaling_cols(data):
    cols=['Price','Marketing_Spend','Discount']
    for col in cols:
        if col in data.columns:
            data[col]=Ss.fit_transform(data[[col]])
    return data

def data_processing(data):
    cols_to_drop=['Revenue','ROI','Date']
    data.drop(columns=[col for col in cols_to_drop if col in data.columns],inplace=True)
    return data

def feature_engineering_workflow(input_path,output_folder,output_file):
    data=pd.read_csv(input_path)
    data=ordinal_data(data)
    data=non_ordinal_data(data)
    data=scaling_cols(data)
    data=data_processing(data)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path=os.path.join(output_folder,output_file)
    data.to_csv(output_path,index=False,encoding='utf-8')
    print(f"Feature-engineered data saved at: {output_path}")
    print(f"Processed Data Summary: Rows-{data.shape[0]}, Columns-{data.shape[1]}")
    return data

if __name__=="__main__":
    input_file="processed_data/cleaned_data.csv"
    output_dir="processed_data/feature_engineered_data"
    output_file_name="feature_engineered_data.csv"
    feature_engineering_workflow(input_path=input_file,output_folder=output_dir,output_file=output_file_name)
