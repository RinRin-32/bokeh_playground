import argparse
import h5py
from bokeh.plotting import curdoc
import numpy as np

from visualizer.decisionboundary import DecisionBoundaryVisualizer
from visualizer.memorymap import MemoryMapVisualizer
from visualizer.sensitivity import SensitivityVisualizer

from bokeh.models import ColumnDataSource

from bokeh.layouts import column, row

import json


# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, required=True, help="Path to the HDF5 file")
args = parser.parse_args()

# Load data from the HDF5 file
with h5py.File(args.file, "r") as f:
    scores_group = f["scores"]
    X = np.array(scores_group["X_train"], dtype=np.float32)
    y = np.array(scores_group["y_train"], dtype=np.int64)
    estimated_deviation = np.array(scores_group["sensitivities"], dtype=np.float64)
    true_deviation = np.array(scores_group["softmax_deviations"], dtype=np.float64)
    bpe = np.array(scores_group["bpe"], dtype=np.float32)
    bls = np.array(scores_group["bls"], dtype=np.float32)

    read = f["config"]["config_data"][()]
    config_json = read.decode("utf-8")
    config = json.loads(config_json)

# Generate IDs
ids = list(range(len(X)))

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

# Create the visualizer instances
decision_boundary_visualizer = DecisionBoundaryVisualizer(shared_source, config)
memory_map_visualizer = MemoryMapVisualizer(shared_source, colors, decision_boundary_visualizer)
sensitivity_visualizer = SensitivityVisualizer(shared_source)

# Create the layout with Memory Map on top left, Decision Boundary on bottom half, and Sensitivity on the right
memory_map_layout = column(memory_map_visualizer.get_layout(), width=400)
decision_boundary_layout = column(decision_boundary_visualizer.get_layout(), height=600)
sensitivity_layout = column(sensitivity_visualizer.get_layout(), width=550)

# Combine the layouts in a row
layout = row(memory_map_layout, decision_boundary_layout, sensitivity_layout)

# Add the combined layout to the Bokeh document
curdoc().add_root(layout)