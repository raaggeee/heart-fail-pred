import numpy as np
from utils.utils import load_dataframe, load_yaml, save_df
from sklearn.model_selection import train_test_split

## open raw file and split data in train and test
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

    save_df("data/interim", "train_data.csv", train_data)
    save_df("data/interim", "test_data.csv", train_data)

if __name__ == "__main__":
    main()

    
