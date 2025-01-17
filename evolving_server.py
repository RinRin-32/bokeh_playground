import pickle

from bokeh.plotting import curdoc

from visualizer.evolvingboundary import EvolvingBoundaryVisualizer
from visualizer.evolvingmpe import EvolvingMemoryMapVisualizer
from visualizer.evolvingsensitivity import EvolvingSensitivityVisualizer

from bokeh.models import ColumnDataSource

from bokeh.layouts import column, row

import numpy as np

dir = 'data/'
file = open(dir + 'evolving_visualizer_evolving_memory_maps_scores.pkl', 'rb')
all_scores = pickle.load(file)
file.close()


file = open(dir + 'evolving_visualizer_evolving_memory_maps_retrain.pkl', 'rb')
retrain = pickle.load(file)
file.close()

X = all_scores[0]['X_train'].numpy()
y = all_scores[0]['y_train'].numpy()
ids = list(range(len(X)))

'''
shared_data = shared_score.data
epoch_index = shared_data["epoch"].index(0)  # Find the index of epoch 0
bpe_value = shared_data["bpe"][epoch_index]
bls_value = shared_data["bls"][epoch_index]

print(f"Shared Score - Epoch 0:\nBPE: {bpe_value}\nBLS: {bls_value}")


retrain_data_dict = retrain_data.data
epoch_index = retrain_data_dict["epoch"].index(0)  # Find the index of epoch 0
sensitivities = retrain_data_dict["sensitivities"][epoch_index]
indices = retrain_data_dict["indices"][epoch_index]
softmax_dev = retrain_data_dict["softmax_deviations"][epoch_index]
'''

shared_source = ColumnDataSource(data={
    "id": ids,
    "x": X[:, 0],
    "y": X[:, 1],
    "class": y,
    "color": ["blue" if cls == 0 else "green" for cls in y],
    "marker": ["circle" if cls == 0 else "square" for cls in y],
})

# Set up classes, colors, and markers
classes = [0, 1]
colors = ["blue", "green"]
markers = ["circle", "square"]

# Extract 'bpe' and 'bls' from all_scores over epochs
bpe_scores = [epoch_data["bpe"] for epoch, epoch_data in all_scores.items()]
bls_scores = [epoch_data["bls"] for epoch, epoch_data in all_scores.items()]
epochs = list(range(len(all_scores)))

shared_score = ColumnDataSource(data={
    "epoch": epochs,
    "bpe": bpe_scores,
    "bls": bls_scores,
    "color": shared_source.data["color"],
    "marker": shared_source.data["marker"],
})

sensitivity_scores = [epoch_data["sensitivities"] for epoch, epoch_data in all_scores.items()]
indices = [epoch_data["indices_retrain"] for epoch, epoch_data in retrain.items()]
softmax_deviation = [epoch_data["softmax_deviations"] for epoch, epoch_data in retrain.items()]

retrain_data = ColumnDataSource(data={
    "epoch": epochs,
    "sensitivities": sensitivity_scores,
    "indices": indices,
    "softmax_deviations": softmax_deviation,
    "color": shared_source.data["color"],
    "marker": shared_source.data["marker"],
})

## NEED TO MAKE COLOR AND MARKER SHARED DATA SOURCE HERE!!!!
# Further headache, how to sync the color for selection
# Making it a class might be easier to sync?
# Can the selected datapoint be tracked over epoch, that'd make a really good plot!


sensitivityvisualizer = EvolvingSensitivityVisualizer(retrain_data)
memorymapvisualzier = EvolvingMemoryMapVisualizer(shared_score)
boundaryvisualizer = EvolvingBoundaryVisualizer(shared_source, sensitivityvisualizer, memorymapvisualzier, 1)

boundary_layout = column(boundaryvisualizer.get_layout(), width=600)
memory_layout = column(memorymapvisualzier.get_layout(), width=600)
sensitivity_layout = column(sensitivityvisualizer.get_layout(), width=600)

layout = row(boundary_layout, memory_layout, sensitivity_layout)

curdoc().add_root(layout)