from bokeh.plotting import curdoc
from bokeh.layouts import column, row
from sklearn.neural_network import MLPClassifier
import numpy as np
from bokeh.models import ColumnDataSource
from visualizer.decisionboundary import DecisionBoundaryVisualizer
from visualizer.memorymap import MemoryMapVisualizer
from visualizer.sensitivity import SensitivityVisualizer

# Generate random 2D data
np.random.seed(42)
n_samples = 200

# Class 0
x0 = np.random.normal(loc=5.0, scale=3.0, size=(n_samples, 2))
y0 = np.zeros(n_samples)

# Class 1
x1 = np.random.normal(loc=-5.0, scale=4.5, size=(n_samples, 2))
y1 = np.ones(n_samples)

# Combine data
X = np.vstack((x0, x1))
y = np.hstack((y0, y1))

# Generate IDs
ids = list(range(len(X)))

np.random.seed(42)  # For reproducibility
estimated_deviation = np.random.rand(len(X))  # Random values between 0 and 1
true_deviation = np.random.rand(len(X))  # Random values between 0 and 1
bpe = np.random.rand(len(X))  # Random values between 0 and 1
bls = np.random.rand(len(X))  # Random values between 0 and 1

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
model = MLPClassifier(hidden_layer_sizes=(50, 50, 100), max_iter=300, random_state=42)

# Set up classes, colors, and markers
classes = [0, 1]
colors = ["blue", "green"]
markers = ["circle", "square"]

# Define the regression line coordinates (replace these with your own coordinates)
line_coords = {
    'x_vals': [0, 0.5, 1],
    'y_vals': [0, 0.5, 1]
}

# Create the visualizer instances
memory_map_visualizer = MemoryMapVisualizer(shared_source, colors)
decision_boundary_visualizer = DecisionBoundaryVisualizer(model, shared_source)
sensitivity_visualizer = SensitivityVisualizer(shared_source, line_coords)

# Create the layout with Memory Map on top left, Decision Boundary on bottom half, and Sensitivity on the right
memory_map_layout = column(memory_map_visualizer.get_layout(), width=400)
decision_boundary_layout = column(decision_boundary_visualizer.get_layout(), height=600)
sensitivity_layout = column(sensitivity_visualizer.get_layout(), width=600)

# Combine the layouts in a row
layout = row(memory_map_layout, decision_boundary_layout, sensitivity_layout)
#layout = row(memory_map_layout, decision_boundary_layout)

# Add the combined layout to the Bokeh document
curdoc().add_root(layout)