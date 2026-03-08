import torch.nn as nn
import torch
import yaml
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np

def load_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        file = yaml.safe_load(f)

    return file

def load_params(yaml_path):
    params = load_yaml(yaml_path)
    return params

def load_dataframe(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    
    except TypeError:
        return "DataFrame not found"
    
    except FileNotFoundError:
        return "Invalid Path"

class ANN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):
        x = torch.relu(self.input(x))

        x = torch.relu(self.hidden(x))
        x = torch.relu(self.hidden(x))

        x = torch.sigmoid(self.output(x))

        return x

def test_model(X):
    model = ANN(input_dim=X.shape[1], hidden_dim=4, output_dim=1)
    X.unsqueeze(0)

    with torch.no_grad():
        preds = (model(X) > 0.5).float()

    return preds
    

def main():
    params = load_params("params.yaml")
    test_df = load_dataframe(params["model_eval"]["test_path"])

    X = torch.tensor(test_df.iloc[:, :-1].values).to(torch.float32)
    y = torch.tensor(test_df.iloc[:, -1].values).to(torch.float32)
    print(X.shape)

    preds = test_model(X)
    print(y, preds)
    print(accuracy_score(np.array(y), np.array(preds)))

if __name__ == "__main__":
    main()


    

    





    




