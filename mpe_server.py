import os
import sys
import pickle
import argparse
import numpy as np

import tqdm

from bokeh.plotting import curdoc

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam

from visualizer.decisionboundary import DecisionBoundaryVisualizer
from visualizer.memorymap import MemoryMapVisualizer
from visualizer.sensitivity import SensitivityVisualizer

from bokeh.models import ColumnDataSource

from bokeh.layouts import column, row

from ivon import IVON as IBLR

sys.path.append("../memory-perturbation")
from lib.models import get_model
from lib.datasets import get_dataset
from lib.utils import get_quick_loader, predict_test, flatten, predict_nll_hess, train_model, predict_train2, train_network
from lib.variances import get_covariance_from_iblr, get_covariance_from_adam, get_pred_vars_optim, get_pred_vars_laplace


dir = 'data/'
file = open(dir + 'moon_large_mlp_epoch20_05_memory_maps_scores.pkl', 'rb')
scores_dict = pickle.load(file)
file.close()


file = open(dir + 'moon_large_mlp_epoch20_05_memory_maps_retrain.pkl', 'rb')
deviation_dict = pickle.load(file)
file.close()

#X,y = make_moons(n_samples=200, noise=5, random_state=42)

X = scores_dict['X_train'].numpy()
y = scores_dict['y_train'].numpy()
estimated_deviation = scores_dict['sensitivities']
# Generate IDs
ids = list(range(len(X)))

softmax = deviation_dict['softmax_deviations']

true_deviation = softmax
bpe = scores_dict['bpe']
bls = scores_dict['bls']

# Shared ColumnDataSource with random values for the new metrics
shared_source = ColumnDataSource(data={
    "id": ids,
    "x": X[:, 0],
    "y": X[:, 1],
    "class": y,
    "color": ["blue" if cls == 0 else "green" for cls in y],
    "marker": ["circle" if cls == 0 else "square" for cls in y],
    "estimated_deviation": estimated_deviation,
    "true_deviation": true_deviation,
    "bpe": bpe,
    "bls": bls
})

# Train a model
#model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=10, random_state=42)
#model = MLPClassifier(hidden_layer_sizes=(500, 300), max_iter=10, random_state=42)
#model = MLPClassifier(hidden_layer_sizes=(500, 300), max_iter=20, random_state=42)

# Set up classes, colors, and markers
classes = [0, 1]
colors = ["blue", "green"]
markers = ["circle", "square"]

# Define the regression line coordinates (replace these with your own coordinates)
line_coords = {
    'x_vals': [0, true_deviation.max()],
    'y_vals': [0, true_deviation.max()]
}

# Create the visualizer instances
memory_map_visualizer = MemoryMapVisualizer(shared_source, colors)
decision_boundary_visualizer = DecisionBoundaryVisualizer(shared_source)
sensitivity_visualizer = SensitivityVisualizer(shared_source, line_coords)

# Create the layout with Memory Map on top left, Decision Boundary on bottom half, and Sensitivity on the right
memory_map_layout = column(memory_map_visualizer.get_layout(), width=400)
decision_boundary_layout = column(decision_boundary_visualizer.get_layout(), height=600)
sensitivity_layout = column(sensitivity_visualizer.get_layout(), width=600)

# Combine the layouts in a row
layout = row(memory_map_layout, decision_boundary_layout, sensitivity_layout)

# Add the combined layout to the Bokeh document
curdoc().add_root(layout)