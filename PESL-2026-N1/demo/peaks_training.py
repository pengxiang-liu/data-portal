# ----------------------------------------------------------------------- #
# MIT license
# Copyright (c) 2026, 
# Data Analytics and Computational Intelligence (DACI) Laboratory,
# Department of Data Science, College of Computing,
# City University of Hong Kong,
# Dr. Pengxiang Liu, All Rights Reserved.
# ----------------------------------------------------------------------- #

"""
Description:
    This script demonstrates the pipeline for constraint learning using 
    Adaptive Sigmoid Partitioning (ASP) algorithm.

Mathematical Formulation:
    min   z
    s.t.  peaks(x, y) == z
          -3 <= x <= 3, -3 <= y <= 3
          x is binary; y and z are continuous

Matrix Formulation:
    min   c * x
    s.t.  A * x == e,
          B * x <= f,
          x[q_n] = 1 / (1 + exp(-x[p_n])), for all (p_n, q_n) in W,
          lb <= x <= ub,
          x_i is binary, for all (i) in Int
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------------------------------------------------- #
# pytorch model
class pytorch_peak(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.out = nn.Linear(8, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return self.out(x)


# ----------------------------------------------------------------------- #
# peaks function
def func_peaks(x, y):
    """Calculates the value of the MATLAB peaks function."""
    z = 3 * (1 - x) ** 2 * np.exp(-(x ** 2) - (y + 1) ** 2) \
        - 10 * (x / 5 - x ** 3 - y ** 5) * np.exp(-x ** 2 - y ** 2) \
        - 1 / 3 * np.exp(-(x + 1) ** 2 - y ** 2)
    return z


# ----------------------------------------------------------------------- #
# visualize peaks function
def visualize_peaks_function(fp_png):
    """Visualizes the peaks function and saves it as a PNG file."""
    n = 100
    x = np.linspace(-3, 3, n)
    y = np.linspace(-3, 3, n)
    x_mesh, y_mesh = np.meshgrid(x, y)
    z_mesh = func_peaks(x_mesh, y_mesh)

    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(111, projection = "3d")
    ax.plot_surface(x_mesh, y_mesh, z_mesh, cmap = "viridis")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    plt.savefig(fp_png, bbox_inches = "tight")
    plt.close()


# ----------------------------------------------------------------------- #
# create dataset
def create_dataset(fp_csv, n_samples = 1000):
    """Creates a dataset based on the peaks function."""
    # generate random samples for x and y
    n = int(np.sqrt(n_samples) * 1.5)
    x = np.linspace(-3, 3, n)
    y = np.linspace(-3, 3, n)
    x_mesh, y_mesh = np.meshgrid(x, y)
    z_mesh = func_peaks(x_mesh, y_mesh)
    # flatten the arrays
    v_flat = [x_mesh.flatten(), y_mesh.flatten(), z_mesh.flatten()]
    # randomly select n_samples
    idx = np.random.choice(len(v_flat[0]), size = n_samples, 
                           replace = False)
    data = {"x": v_flat[0][idx], "y": v_flat[1][idx], "z": v_flat[2][idx]}
    df = pd.DataFrame(data)
    df = df.round(4)
    df.to_csv(fp_csv, index = False)


# ----------------------------------------------------------------------- #
# train neural approximation
def train_neural_network(fp_csv, fp_net):
    """Trains a neural network to approximate the peaks function."""
    # load the dataset
    col_x, col_y = ["x", "y"], ["z"]
    df = pd.read_csv(fp_csv, usecols = col_x + col_y)
    df_x, df_y = df[col_x], df[col_y]
    # scale the data
    df_x = (df_x - df_x.mean()).divide(df_x.std())
    df_y = (df_y - df_y.mean()).divide(df_y.std())
    # convert to numpy arrays
    x = df_x.to_numpy()
    y = df_y.to_numpy()

    # split the dataset into training and validation sets
    n_samples = x.shape[0]
    n_train = int(n_samples * 0.8)
    x_train, y_train = x[:n_train], y[:n_train]
    x_valid, y_valid = x[n_train:], y[n_train:]

    # create data loaders
    batch_size = 32
    train_dataset = TensorDataset(
        torch.as_tensor(x_train, dtype = torch.float32),
        torch.as_tensor(y_train, dtype = torch.float32)
    )
    valid_dataset = TensorDataset(
        torch.as_tensor(x_valid, dtype = torch.float32),
        torch.as_tensor(y_valid, dtype = torch.float32)
    )
    train_loader = DataLoader(train_dataset, batch_size = batch_size, 
                              shuffle = True)
    valid_loader = DataLoader(valid_dataset, batch_size = batch_size, 
                              shuffle = False)
    # create the model
    input_dim, output_dim = len(col_x), len(col_y)
    model = pytorch_peak(input_dim = input_dim, output_dim = output_dim)

    # define the loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.01)
    # training loop
    n_epochs = 1000
    for epoch in range(n_epochs):
        # training
        model.train()
        for x_batch, y_batch in train_loader:
            y_pred = model(x_batch)
            loss = loss_function(y_pred, y_batch.view(*y_pred.shape))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # validation
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in valid_loader:
                y_pred = model(x_batch)
                loss = loss_function(y_pred, y_batch.view(*y_pred.shape))
                valid_loss += loss.item() * x_batch.size(0)
        valid_loss /= len(valid_loader.dataset)
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, "
                  f"Training Loss: {loss.item():.6f}, "
                  f"Validation Loss: {valid_loss:.6f}")
    
    # export the trained model to ONNX format
    x_dummy = torch.randn(batch_size, input_dim, requires_grad = True)
    torch.onnx.export(
        model, x_dummy, fp_net,
        input_names = ["input"], output_names = ["output"],
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )


# ----------------------------------------------------------------------- #
# main function
if __name__ == "__main__":

    # set the paths
    fp_dir = os.path.dirname(os.path.abspath(__file__))
    fp_png = os.path.join(fp_dir, "peaks_function.png")
    fp_csv = os.path.join(fp_dir, "peaks_data.csv")
    fp_net = os.path.join(fp_dir, "peaks_neural_approx.onnx")

    # data visualization
    if not os.path.exists(fp_png):
        visualize_peaks_function(fp_png)
    # data-preparation
    if not os.path.exists(fp_csv):
        create_dataset(fp_csv, n_samples = 1000)
    # data training
    if not os.path.exists(fp_net):
        train_neural_network(fp_csv, fp_net)