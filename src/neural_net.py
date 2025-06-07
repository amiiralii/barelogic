import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from bl import *
import warnings

# Suppress CUDA warning
warnings.filterwarnings("ignore", message=".*CUDA initialization.*")

def neural_net(X_test, y_test, X_train, y_train, cols):
    features = [c for c in cols if c[-1] not in ["+","-", "X"]]
    numerical_cols = [f for f in features if f[0].isupper()]
    categorical_cols = [f for f in features if not f[0].isupper()]
    
    X_train.drop([c for c in cols if c[-1] in ["+","-", "X"]], axis=1, inplace=True)
    X_test.drop([c for c in cols if c[-1] in ["+","-", "X"]], axis=1, inplace=True)

    # Preprocessing: Encode categorical + scale numerical
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)


    # Convert DataFrame to numpy array first
    X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    # Define simple feed-forward neural network
    class SimpleRegressor(nn.Module):
        def __init__(self, input_dim):
            super(SimpleRegressor, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)  # output a single value
            )

        def forward(self, x):
            return self.model(x)

    # Instantiate model
    model = SimpleRegressor(input_dim=X_train_tensor.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()
        #print(f"Epoch {epoch+1}: Train Loss = {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        test_preds = [t[0] for t in model(X_test_tensor).tolist()]
        return test_preds
