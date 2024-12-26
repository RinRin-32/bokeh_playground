from sklearn.neural_network import MLPClassifier
import numpy as np
from bokeh.plotting import curdoc
from decisionboundary import DecisionBoundaryVisualizer


n_samples = 200

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

model = MLPClassifier(hidden_layer_sizes=(50, 50, 100), max_iter=300, random_state=42)
classes = [0, 1]
colors = ["blue", "green"]
markers = ["circle", "square"]

visualizer = DecisionBoundaryVisualizer(model, X, y, classes, colors, markers)
curdoc().add_root(visualizer.layout())