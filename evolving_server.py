import pickle

from bokeh.plotting import curdoc

from visualizer.evolvingboundary import EvolvingBoundaryVisualizer

from bokeh.models import ColumnDataSource

import numpy as np

dir = 'data/'
file = open(dir + 'test_evolve_small_loo_evolving_memory_maps_scores.pkl', 'rb')
all_scores = pickle.load(file)
file.close()


file = open(dir + 'test_evolve_small_loo_evolving_memory_maps_retrain.pkl', 'rb')
retrain = pickle.load(file)
file.close()

X = all_scores[0]['X_train'].numpy()
y = all_scores[0]['y_train'].numpy()
ids = list(range(len(X)))

# Extract 'bpe' and 'bls' from all_scores over epochs
bpe_scores = [epoch_data["bpe"] for epoch, epoch_data in all_scores.items()]
bls_scores = [epoch_data["bls"] for epoch, epoch_data in all_scores.items()]
epochs = list(range(len(all_scores)))

shared_score = ColumnDataSource(data={
    "epoch": epochs,
    "bpe": bpe_scores,
    "bls": bls_scores
})

sensitivity_scores = [epoch_data["sensitivities"] for epoch, epoch_data in all_scores.items()]
indices = [epoch_data["indices_retrain"] for epoch, epoch_data in retrain.items()]
softmax_deviation = [epoch_data["softmax_deviations"] for epoch, epoch_data in retrain.items()]

retrain_data = ColumnDataSource(data={
    "epoch": epochs,
    "sensitivities": sensitivity_scores,
    "indices": indices,
    "softmax_deviations": softmax_deviation
})


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

visualizer = EvolvingBoundaryVisualizer(shared_source)

curdoc().add_root(visualizer.get_layout())