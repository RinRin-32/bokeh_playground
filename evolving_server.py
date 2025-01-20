import pickle

from bokeh.plotting import curdoc

from visualizer.evolvingboundary import EvolvingBoundaryVisualizer
from visualizer.evolvingmpe import EvolvingMemoryMapVisualizer
from visualizer.evolvingsensitivity import EvolvingSensitivityVisualizer

from bokeh.models import ColumnDataSource

from bokeh.layouts import column, row

import numpy as np

dir = 'data/'
#file = open(dir + 'evolving_visualizer_evolving_memory_maps_scores.pkl', 'rb')
file = open(dir + 'visualize_evolving_evolving_memory_maps_scores.pkl', 'rb')
all_scores = pickle.load(file)
file.close()


#file = open(dir + 'evolving_visualizer_evolving_memory_maps_retrain.pkl', 'rb')
file = open(dir + 'visualize_evolving_evolving_memory_maps_retrain.pkl', 'rb')
retrain = pickle.load(file)
file.close()

X = all_scores[0]['X_train'].numpy()
y = all_scores[0]['y_train'].numpy()
ids = list(range(len(X)))

# Set up classes, colors, and markers
classes = [0, 1]
colors = ["blue", "green"]
markers = ["circle", "square"]

# Extract 'bpe' and 'bls' from all_scores over epochs
bpe_scores = [epoch_data["bpe"] for epoch, epoch_data in all_scores.items()]
bls_scores = [epoch_data["bls"] for epoch, epoch_data in all_scores.items()]
epochs = list(range(len(all_scores)))
sensitivity_scores = [epoch_data["sensitivities"] for epoch, epoch_data in all_scores.items()]
indices = [epoch_data["indices_retrain"] for epoch, epoch_data in retrain.items()]
softmax_deviation = [epoch_data["softmax_deviations"] for epoch, epoch_data in retrain.items()]

xx = [scores['decision_boundary']['xx'] for epoch, scores in all_scores.items()]
yy = [scores['decision_boundary']['yy'] for epoch, scores in all_scores.items()]
Z = [scores['decision_boundary']['Z'] for epoch, scores in all_scores.items()]

X_train = [scores['X_train'] for epoch, scores in all_scores.items()]
y_train = [scores['y_train'] for epoch, scores in all_scores.items()]


shared_source = ColumnDataSource(data={
    "id": ids,
    "x": X[:, 0],
    "y": X[:, 1],
    "class": y,
    "color": ["blue" if cls == 0 else "green" for cls in y],
    "marker": ["circle" if cls == 0 else "square" for cls in y],
    "bpe": bpe_scores[0],
    "bls": bls_scores[0],
    "sensitivities": sensitivity_scores[0],
    "softmax_deviations": softmax_deviation[0],
})

shared_resource = ColumnDataSource(data={
    "epoch": epochs,
    "xx": xx,
    "yy": yy,
    "Z": Z,
    "bpe": bpe_scores,
    "bls": bls_scores,
    "sensitivities": sensitivity_scores,
    "softmax_deviations": softmax_deviation,
})

## NEED TO MAKE COLOR AND MARKER SHARED DATA SOURCE HERE!!!!
# Further headache, how to sync the color for selection
# Making it a class might be easier to sync?
# Can the selected datapoint be tracked over epoch, that'd make a really good plot!


sensitivityvisualizer = EvolvingSensitivityVisualizer(shared_source)
memorymapvisualzier = EvolvingMemoryMapVisualizer(shared_source)
boundaryvisualizer = EvolvingBoundaryVisualizer(shared_source, shared_resource, sensitivityvisualizer, 1, colors)

boundary_layout = column(boundaryvisualizer.get_layout(), width=600)
memory_layout = column(memorymapvisualzier.get_layout(), width=600)
sensitivity_layout = column(sensitivityvisualizer.get_layout(), width=600)

layout = row(boundary_layout, memory_layout, sensitivity_layout)

curdoc().add_root(layout)