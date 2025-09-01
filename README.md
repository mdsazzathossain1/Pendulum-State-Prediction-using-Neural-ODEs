# Pendulum-State-Prediction-using-Neural-ODEs
This notebook demonstrates the use of Neural Ordinary Differential Equations (Neural ODEs) to predict the state variables of a pendulum (such as angle and angular velocity). It leverages PyTorch, PyTorch Lightning, and Torchdiffeq to build, train, and evaluate models that approximate the dynamics of physical systems.\
Key Features:

Data preprocessing & pendulum simulation data handling.

Neural ODE implementation with PyTorch.

Training with PyTorch Lightning + early stopping.

Prediction of future pendulum trajectories.

Visualization of actual vs predicted dynamics.

Animated plots for trajectory evolution.

🔑 Step-by-Step Breakdown

Setup & Libraries

Uses torch, torchdiffeq, pytorch_lightning, numpy, matplotlib.

Installs additional libraries (torchdyn, torchdiffeq).

Data loading and saving handled with pickle.

Data Loading & Preprocessing

Loads pendulum trajectory data (angle & velocity) from pickle files.

Prepares time series data for training and testing.

Converts data into PyTorch Tensors and wraps with TensorDataset & DataLoader.

Model Definition (Neural ODE)

Defines an ODE function (ODEFunc) with a neural network inside.

Uses torchdiffeq’s odeint and odeint_adjoint to integrate the learned dynamics over time.

Models the pendulum’s future states from initial conditions.

Training the Model

Trains using PyTorch Lightning for modularity and early stopping.

Loss function: Mean Squared Error (MSE).

Optimizer: Adam.

Tracks training and validation loss.

Prediction & Visualization

Predicts future pendulum states given past trajectory.

Plots:

Actual vs predicted pendulum angle and velocity.

Future trajectory extrapolation beyond training range.

Includes animated visualization of real vs predicted trajectories.

Advanced Features

Compares results with different hidden dimensions and architectures.

Tests generalization by predicting beyond training horizon.

Uses interpolation for smoother trajectories.
