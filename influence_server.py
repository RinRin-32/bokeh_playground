import h5py
from bokeh.plotting import curdoc
from bokeh.models import ColumnDataSource
from bokeh.layouts import column, row
import json
import sys
import argparse
import os
from skimage import measure
from bokeh.plotting import output_file, save
import numpy as np
from visualizer.influence_snap import LSBoundaryVisualizer

def extract_boundary_lines(xx, yy, zz):
    contours = measure.find_contours(zz, level=0.5)
    xs, ys = [], []
    for contour in contours:
        xs.append(xx[0, 0] + contour[:, 1] * (xx[0, -1] - xx[0, 0]) / zz.shape[1])
        ys.append(yy[0, 0] + contour[:, 0] * (yy[-1, 0] - yy[0, 0]) / zz.shape[0])
    return xs, ys

parser = argparse.ArgumentParser(description="Launch the Bokeh server with an HDF5 file, this plot is to display changes in model behavior over training step.")
parser.add_argument("--file", type=str, required=True, help="Path to the HDF5 file")
parser.add_argument("--output", type=str, required=False, help="If specified filename, while running on python not bokeh serve, the html will be saved in ./output")
parser.add_argument("--scale_factor", type=int, default=3, help="Scale plotting of influence exponentially, default set at 3")

args = parser.parse_args()

h5_file = args.file

if not h5_file.lower().endswith(".h5"):
    print(f"Error: The input file '{h5_file}' is not an HDF5 (.h5) file.")
    sys.exit(1)

if not os.path.isfile(h5_file):
    print(f"Error: The file '{h5_file}' does not exist.")
    sys.exit(1)

if args.output is not None:
    os.makedirs('./output', exist_ok=True)
    output_file(filename=f"./output/{args.output}.html", title="Static HTML file", mode="inline")

with h5py.File(h5_file, "r") as f:
    read = f["config"]["config_data"][()]
    config_json = read.decode("utf-8")
    config = json.loads(config_json)
    dataset = config.get("dataset")
    max_epoch = config.get("max_epochs")
    total_batches = config.get("total_batch")
    max_step = total_batches * max_epoch
    bs = config.get("batch_size")

    X_coord = np.array(f["coord/X_train"])
    y_train = np.array(f["coord/y_train"])

    step_update = [f[f"scores/step_{epoch}"]["param_update"][()] for epoch in range(max_step)]

    xx = [f[f"scores/step_{epoch}"]["decision_boundary"]["xx"][:] for epoch in range(max_step)]
    yy = [f[f"scores/step_{epoch}"]["decision_boundary"]["yy"][:] for epoch in range(max_step)]
    Z = [f[f"scores/step_{epoch}"]["decision_boundary"]["Z"][:] for epoch in range(max_step)]

colors = ["white", "white"]
marker = ["circle", "star"]

temp = []  # Initialize temp as a list to store the norms
for step in range(len(step_update)):
    norm = np.linalg.norm(step_update[step])
    temp.append(norm)

step_update = temp 

param_update = []
temp = []

for step in range(len(step_update)):
    if step % total_batches == 0 and step > 0 or step == max_step-1:
        param_update.append(temp)
        temp = []
    temp.append(step_update[step])
#print(len(param_update), len(param_update[0]))

xs = []
ys = []
for step in range(max_step):
    if ((step+1) % total_batches == 0 and step>0) or step == max_step-1:
        xx_step = xx[step]
        yy_step = yy[step]
        zz_step = Z[step]
        
        boundary_x, boundary_y = extract_boundary_lines(xx_step, yy_step, zz_step)
        xs.append(boundary_x)
        ys.append(boundary_y)

scaled_alphas_list = []
scaled_sizes_list = []
alpha_min, alpha_max = 0.2, 1.0
size_min, size_max = 5, 50
scaling_factor = args.scale_factor  # Adjust to control exaggeration

for epoch_noises in param_update:
    normed_values = (epoch_noises - np.min(epoch_noises)) / (np.max(epoch_noises) - np.min(epoch_noises) + 1e-8)
    
    # Apply exponential transformation to exaggerate differences
    exp_values = normed_values ** scaling_factor  

    alpha_assignments = alpha_min + (alpha_max - alpha_min) * exp_values
    size_assignments = size_min + (size_max - size_min) * exp_values

    scaled_alphas_list.append(alpha_assignments.tolist())
    scaled_sizes_list.append(size_assignments.tolist())

shared_resource = ColumnDataSource(data={
    "epoch": list(range(max_step//total_batches)),
    "xs": xs,
    "ys": ys,
    "size": scaled_sizes_list,
    "alpha": scaled_alphas_list,
})

shared_source = ColumnDataSource(data={
    "x": X_coord[:, 0],
    "y": X_coord[:, 1],
    "class": y_train,
    "color": ['white'] * len(y_train),
    "marker": [marker[cls] for cls in y_train],
    "size": scaled_sizes_list[0],
    "alpha": scaled_alphas_list[0]
})

boundary = LSBoundaryVisualizer(shared_source, shared_resource, max_epoch-1, colors, mode='Epoch')

boundary_layout = column(boundary.get_layout(), sizing_mode="scale_both")

layout = row(boundary_layout)

curdoc().add_root(layout)

if args.output is not None:
    layout.sizing_mode = "scale_both"
    save(layout)