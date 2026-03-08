import torch.nn as nn
import torch
import yaml
import pandas as pd


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

        x = self.output(x)

        return x


def fit(input_dim, output_dim, lr, epochs, batch_size, X, y, iter, hidden_dim=4):
    model = ANN(input_dim, hidden_dim, output_dim)

    criterion = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    n_sample = X.shape[0]

    for j in range(epochs):
        print(f"====EPOCH-{j}====")
        for i in range(0, n_sample, batch_size):
            optim.zero_grad()
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size].unsqueeze(1)
            pred = model(X_batch)
            # print(y_batch)
            # print(pred)

            loss = criterion(pred, y_batch)

            loss.backward()
            optim.step()
            print(f"Loss: {loss.item()}")

    torch.save(model.state_dict(), f"models/model.pkl")

def main():
    params = load_params("params.yaml")
    train_data = load_dataframe(params["model"]["train_path"])
    
    X = train_data.iloc[:, :-1]
    y = train_data.iloc[:, -1]

    X_torch = torch.tensor(X.values).to(torch.float32)
    y_torch = torch.tensor(y.values).to(torch.float32)

    input_dims = int(X_torch.shape[1])
    print(X_torch.shape)
    output_dims = 1

    fit(
        input_dims,
        output_dims,
        lr = params["model"]["optmizers"]["learning_rate"],
        epochs = params["model"]["epochs"],
        batch_size=params["model"]["batch_size"],
        X=X_torch,
        y=y_torch,
        iter=params["model"]["iter"]
    )

if __name__ == "__main__":
    main()


    

        



        

        
