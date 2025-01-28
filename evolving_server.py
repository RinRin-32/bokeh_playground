import pickle

from bokeh.plotting import curdoc

from visualizer.evolvingboundary import EvolvingBoundaryVisualizer
from visualizer.evolvingmpe import EvolvingMemoryMapVisualizer
from visualizer.evolvingsensitivity import EvolvingSensitivityVisualizer

from bokeh.models import ColumnDataSource

from bokeh.layouts import column, row

import numpy as np

dir = 'data/'
#file = open(dir + 'fix_evolving_memory_maps_scores.pkl', 'rb')
file = open(dir + 'debug_evolving_memory_maps_scores.pkl', 'rb')
#file = open(dir + 'debug_scores.pkl', 'rb')
all_scores = pickle.load(file)
file.close()

#file = open(dir + 'fix_evolving_memory_maps_retrain.pkl', 'rb')
file = open(dir + 'debug_evolving_memory_maps_retrain.pkl', 'rb')
#file = open(dir + 'debug_retrain.pkl', 'rb')
retrain = pickle.load(file)
file.close()

X = all_scores[0]['X_train'].numpy()
y = all_scores[0]['y_train'].numpy()
ids = list(range(len(X)))

colors = ["blue", "green"]

# Extract 'bpe' and 'bls' from all_scores over steps
bpe_scores = [step_data["bpe"] for step, step_data in all_scores.items()]
bls_scores = [step_data["bls"] for step, step_data in all_scores.items()]
steps = list(range(len(all_scores)))
sensitivity_scores = [step_data["sensitivities"] for step, step_data in all_scores.items()]
indices = [step_data["indices_retrain"] for step, step_data in retrain.items()]
softmax_deviation = [step_data["softmax_deviations"] for step, step_data in retrain.items()]
print(len(softmax_deviation))

xx = [scores['decision_boundary']['xx'] for step, scores in all_scores.items()]
yy = [scores['decision_boundary']['yy'] for step, scores in all_scores.items()]
Z = [scores['decision_boundary']['Z'] for step, scores in all_scores.items()]

X_train = [scores['X_train'] for step, scores in all_scores.items()]
y_train = [scores['y_train'] for step, scores in all_scores.items()]

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
    "step": steps,
    "xx": xx,
    "yy": yy,
    "Z": Z,
    "bpe": bpe_scores,
    "bls": bls_scores,
    "sensitivities": sensitivity_scores,
    "softmax_deviations": softmax_deviation,
})

sensitivityvisualizer = EvolvingSensitivityVisualizer(shared_source)
memorymapvisualzier = EvolvingMemoryMapVisualizer(shared_source)
boundaryvisualizer = EvolvingBoundaryVisualizer(shared_source, shared_resource, sensitivityvisualizer, 1, colors, max_steps=123)

boundary_layout = column(boundaryvisualizer.get_layout(), width=600)
memory_layout = column(memorymapvisualzier.get_layout(), width=600)
sensitivity_layout = column(sensitivityvisualizer.get_layout(), width=450)

layout = row(boundary_layout, memory_layout, sensitivity_layout)

curdoc().add_root(layout)