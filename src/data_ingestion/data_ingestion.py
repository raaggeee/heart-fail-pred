import numpy as np
# from src.utils.utils_modules import load_dataframe, load_yaml, save_df
from sklearn.model_selection import train_test_split
import os
import yaml 
import pandas as pd
## open raw file and split data in train and test

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
    
def save_df(file_name, df, file_path="/data2/heaart-fail-prediction/data/interim"):
    os.makedirs(file_path, exist_ok=True)
    os.chdir(file_path)
    df.to_csv(file_name)

def params(yaml_path):
    yaml_params = load_yaml(yaml_path)
    test_size = yaml_params["data_ingestion"]["test_size"]
    df_path = yaml_params["data_ingestion"]["file_path"]
    return test_size, df_path

def split_file(file_path, test_size):
    df = load_dataframe(file_path)
    train_data, test_size = train_test_split(df, test_size=0.2, random_state=42)
    return train_data, test_size

def main():
    test_size, df_path = params("params.yaml")
    train_data, test_data = split_file(df_path, test_size)
    save_df("train_data.csv", train_data)
    save_df("test_data.csv", test_data)

if __name__ == "__main__":
    main()

    
