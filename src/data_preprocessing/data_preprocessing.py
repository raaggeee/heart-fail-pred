import pandas as pd
import numpy as np
import yaml
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

#change the Sex, ChestPainType, RestingECG -> encodings
os.makedirs("data/preprocessed", exist_ok=True)
def load_yaml(file_path):
    with open(file_path, "r") as f:
        file = yaml.safe_load(f)

    return file

def load_dataframe(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    
    except TypeError:
        return "DataFrame not found"
    
    except FileNotFoundError:
        return "Invalid Path"

def load_params(yaml_path):
    params = load_yaml(yaml_path)
    train_path = params["data_preprocessing"]["train_path"]
    test_path = params["data_preprocessing"]["test_path"]
    return train_path, test_path

def label_encode(df, type, encoders):
    if type == "train":
        for col in df.columns:
            if df[col].dtype == "str":
                lable_encoder = LabelEncoder()
                df[col] = lable_encoder.fit_transform(df[col])
                encoders[col] = lable_encoder
        return df
    
    else:
        for col in df.columns:
            if col in encoders:
                df[col] = encoders[col].transform(df[col])
        return df
    

def save_df(file_name, df, file_path="data/preprocessed"):
    print("creating..")
    os.makedirs(file_path, exist_ok=True)
    df.to_csv(os.path.join(file_path,file_name), index=False)

def main():
    scaler = StandardScaler()
    encoder = {}

    train_path, test_path = load_params("params.yaml")

    train_data = load_dataframe(train_path).iloc[:, :-1]
    train_res = load_dataframe(train_path).iloc[:,-1]
    test_data = load_dataframe(test_path).iloc[:, :-1]
    test_res = load_dataframe(test_path).iloc[:, -1]

    print(train_data.dtypes)

    train_data_label = label_encode(train_data, "train", encoder)
    test_data_label = label_encode(test_data, "test", encoder)

    train_data_scale = scaler.fit_transform(train_data_label)
    test_data_scale = scaler.transform(test_data_label)

    train_final = pd.DataFrame(train_data_scale)
    train_final[12] = train_res
    test_final = pd.DataFrame(test_data_scale)
    test_final[12] = test_res

    save_df("train_data_processed.csv", train_final)
    save_df("test_data_processed.csv", test_final)

if __name__ == "__main__":
    main()






    