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
from visualizer.evolvingboundary import EvolvingBoundaryVisualizer

from bokeh.models import ColumnDataSource

from bokeh.layouts import column, row

from ivon import IVON as IBLR

sys.path.append("../memory-perturbation")
from lib.models import get_model
from lib.datasets import get_dataset
from lib.utils import get_quick_loader, predict_test, flatten, predict_nll_hess, train_model, predict_train2, train_network
from lib.variances import get_covariance_from_iblr, get_covariance_from_adam, get_pred_vars_optim, get_pred_vars_laplace

dir = 'data/'
#file = open(dir + 'nonlinear_moon_50_memory_maps_scores.pkl', 'rb')
file = open(dir + '30_epoch_mlp_memory_maps_scores.pkl', 'rb')
scores_dict = pickle.load(file)
file.close()


#file = open(dir + 'nonlinear_moon_50_memory_maps_retrain.pkl', 'rb')
file = open(dir + '30_epoch_mlp_memory_maps_retrain.pkl', 'rb')
deviation_dict = pickle.load(file)
file.close()

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

# Set up classes, colors, and markers
classes = [0, 1]
colors = ["blue", "green"]
markers = ["circle", "square"]

# Define the regression line coordinates (replace these with your own coordinates)
line_coords = {
    'x_vals': [0, true_deviation.max()],
    'y_vals': [0, true_deviation.max()]
}

visualizer = EvolvingBoundaryVisualizer(shared_source)

curdoc().add_root(visualizer.get_layout())