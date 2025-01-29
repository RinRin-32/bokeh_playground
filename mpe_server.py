import pickle

from bokeh.plotting import curdoc

from visualizer.decisionboundary import DecisionBoundaryVisualizer
from visualizer.memorymap import MemoryMapVisualizer
from visualizer.sensitivity import SensitivityVisualizer

from bokeh.models import ColumnDataSource

from bokeh.layouts import column, row


dir = 'data/'
#file = open(dir + 'nonlinear_moon_50_memory_maps_scores.pkl', 'rb')
file = open(dir + '30_epoch_mlp_memory_maps_scores.pkl', 'rb')
scores_dict = pickle.load(file)
file.close()


#file = open(dir + 'nonlinear_moon_50_memory_maps_retrain.pkl', 'rb')
file = open(dir + '30_epoch_mlp_memory_maps_retrain.pkl', 'rb')
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

# Set up classes, colors, and markers
classes = [0, 1]
colors = ["blue", "green"]
markers = ["circle", "square"]

# Create the visualizer instances
decision_boundary_visualizer = DecisionBoundaryVisualizer(shared_source)
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