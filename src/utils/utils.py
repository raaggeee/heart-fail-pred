import pandas as pd
import yaml
import os

def load_yaml(file_path):
    with open(file_path, "r") as f:
        file = yaml.safe_load()

    return file

def load_dataframe(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    
    except TypeError:
        return "DataFrame not found"
    
    except FileNotFoundError:
        return "Invalid Path"
    
def save_df(file_path, file_name, df):
    os.chdir(file_path)
    df.to_csv(file_name)
