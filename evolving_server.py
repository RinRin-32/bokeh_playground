import h5py
from bokeh.plotting import curdoc
from visualizer.evolvingboundary import EvolvingBoundaryVisualizer
from visualizer.evolvingmpe import EvolvingMemoryMapVisualizer
from visualizer.evolvingsensitivity import EvolvingSensitivityVisualizer
from bokeh.models import ColumnDataSource
from bokeh.layouts import column, row
import json
import sys
import argparse
import os
from skimage import measure
from bokeh.plotting import output_file, save

def extract_boundary_lines(xx, yy, zz):
    contours = measure.find_contours(zz, level=0.5)  # Assuming boundary at 0.5 probability
    xs, ys = [], []
    for contour in contours:
        xs.append(xx[0, 0] + contour[:, 1] * (xx[0, -1] - xx[0, 0]) / zz.shape[1])
        ys.append(yy[0, 0] + contour[:, 0] * (yy[-1, 0] - yy[0, 0]) / zz.shape[0])
    return xs, ys

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Launch the Bokeh server with an HDF5 file, this plot is to display changes in model behavior over training step.")
parser.add_argument("--file", type=str, required=True, help="Path to the HDF5 file")
parser.add_argument("--output", type=str, required=False, help="If specified filename, while running on python not bokeh serve, the html will be saved in ./output")
args = parser.parse_args()

# Load the HDF5 file
h5_file = args.file

# Check if the file has an .h5 extension
if not h5_file.lower().endswith(".h5"):
    print(f"Error: The input file '{h5_file}' is not an HDF5 (.h5) file.")
    sys.exit(1)

# Check if the file exists
if not os.path.isfile(h5_file):
    print(f"Error: The file '{h5_file}' does not exist.")
    sys.exit(1)

if args.output is not None:
    os.makedirs('./output', exist_ok=True)
    output_file(filename=f"./output/{args.output}.html", title="Static HTML file", mode="inline")

with h5py.File(h5_file, "r") as f:
    # Read config (if needed for any parameters, e.g., max_steps)
    read = f["config"]["config_data"][()]
    config_json = read.decode("utf-8")
    config = json.loads(config_json)
    total_steps = config.get("total_step")
    log_step = config.get("log_step")
    total_batch = config.get("total_batch")
    total_epoch = config.get("epoch")
    
    # Read X_train and y_train from coord
    X_train = f["coord"]["X_train"][:]
    y_train = f["coord"]["y_train"][:]
    ids = list(range(len(X_train)))

    # Extract data from scores group
    bpe_scores = [f[f"scores/step_{step}"]["bpe"][()] for step in range(total_steps)]
    bls_scores = [f[f"scores/step_{step}"]["bls"][()] for step in range(total_steps)]
    softmax_deviation = [f[f"scores/step_{step}"]["softmax_deviations"][()] for step in range(total_steps)]

    # Extract decision boundary data
    xx = [f[f"scores/step_{step}"]["decision_boundary"]["xx"][:] for step in range(total_steps)]
    yy = [f[f"scores/step_{step}"]["decision_boundary"]["yy"][:] for step in range(total_steps)]
    Z = [f[f"scores/step_{step}"]["decision_boundary"]["Z"][:] for step in range(total_steps)]

    # Sensitivity scores (if part of the scores group)
    sensitivity_scores = [f[f"scores/step_{step}"]["sensitivities"][:] for step in range(total_steps)]

# Define colors and markers based on class
colors = ["blue", "green"]
marker = ["circle", "square"]


xs = []
ys = []
for step in range(total_steps):
    xx_step = xx[step]
    yy_step = yy[step]
    zz_step = Z[step]
    
    # Extract boundary for each step
    boundary_x, boundary_y = extract_boundary_lines(xx_step, yy_step, zz_step)
    xs.append(boundary_x)
    ys.append(boundary_y)

shared_resource = ColumnDataSource(data={
    "step": list(range(total_steps)),
    "bpe": bpe_scores,
    "bls": bls_scores,
    "sensitivities": sensitivity_scores,
    "softmax_deviations": softmax_deviation,
    "xs": xs,
    "ys": ys,
})

# Prepare the shared sources
shared_source = ColumnDataSource(data={
    "id": ids,
    "x": X_train[:, 0],  # First dimension of X_train
    "y": X_train[:, 1],  # Second dimension of X_train
    "class": y_train,  # Class labels
    "color": [colors[cls] for cls in y_train],
    "marker": [marker[cls] for cls in y_train],
    "alpha": [1.0] * len(y_train),
    "size": [6] * len(y_train),
    "bpe": bpe_scores[0],
    "bls": bls_scores[0],
    "sensitivities": sensitivity_scores[0],
    "softmax_deviations": softmax_deviation[0],
})

# Initialize visualizers
sensitivityvisualizer = EvolvingSensitivityVisualizer(shared_source)
memorymapvisualizer = EvolvingMemoryMapVisualizer(shared_source)
boundaryvisualizer = EvolvingBoundaryVisualizer(
    shared_source,
    shared_resource,
    log_step,
    colors,
    total_batch,
    max_steps=total_steps - 1
)

# Layout
boundary_layout = column(boundaryvisualizer.get_layout(), width=575, height=575)
memory_layout = column(memorymapvisualizer.get_layout(), width=600)
sensitivity_layout = column(sensitivityvisualizer.get_layout(), width=450)

layout = row(boundary_layout, memory_layout, sensitivity_layout)

curdoc().add_root(layout)

if args.output is not None:
    save(layout)