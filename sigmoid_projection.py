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
from visualizer.ls_decisionboundary import LSBoundaryVisualizer
from visualizer.projection import ProjectionPlot
from visualizer.noise_bar import BarProjectionPlot


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

    X_coord = np.array(f["coord/X_train"])
    y_train = np.array(f["coord/y_train"])

    all_epoch_noises = [f[f"scores/step_{epoch}"]["noise"][()] for epoch in range(max_step)]
    logits = [f[f"scores/step_{epoch}"]["logits"][()] for epoch in range(max_step)]
    sig_in = [f[f"scores/step_{epoch}"]["sig_input"][()] for epoch in range(max_step)]

    xx = [f[f"scores/step_{epoch}"]["decision_boundary"]["xx"][:] for epoch in range(max_step)]
    yy = [f[f"scores/step_{epoch}"]["decision_boundary"]["yy"][:] for epoch in range(max_step)]
    Z = [f[f"scores/step_{epoch}"]["decision_boundary"]["Z"][:] for epoch in range(max_step)]

colors = ["white", "white"]
marker = ["circle", "square"]

xs = []
ys = []
for step in range(max_step):
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

for epoch_noises in all_epoch_noises:
    normed_values = (epoch_noises - np.min(epoch_noises)) / (np.max(epoch_noises) - np.min(epoch_noises) + 1e-8)
    
    # Apply exponential transformation to exaggerate differences
    exp_values = normed_values ** scaling_factor  

    alpha_assignments = alpha_min + (alpha_max - alpha_min) * exp_values
    size_assignments = size_min + (size_max - size_min) * exp_values

    scaled_alphas_list.append(alpha_assignments.tolist())
    scaled_sizes_list.append(size_assignments.tolist())

num_regions = 24  # Define a fixed number of regions
region_edges = np.linspace(np.min(sig_in), np.max(sig_in), num_regions + 1)
region_centers = (region_edges[:-1] + region_edges[1:]) / 2  # Fixed bin centers

region_means_list = []
region_sig_in_list = []

for epoch_noises, sig_in_epoch in zip(all_epoch_noises, sig_in):
    region_means = np.zeros(num_regions)  # Initialize all bins to zero
    region_counts = np.zeros(num_regions)  # Track number of points per bin

    for i in range(num_regions):
        region_mask = (sig_in_epoch >= region_edges[i]) & (sig_in_epoch < region_edges[i + 1])
        
        if np.any(region_mask):
            region_means[i] = np.mean(epoch_noises[region_mask])
            region_counts[i] = np.sum(region_mask)  # Count the number of points in this bin

    region_means_list.append(region_means)
    region_sig_in_list.append(region_centers)


shared_resource = ColumnDataSource(data={
    "epoch": list(range(max_step)),
    "xs": xs,
    "ys": ys,
    "size": scaled_sizes_list,
    "alpha": scaled_alphas_list,
    "sig_in": sig_in,
    "logits": logits,
    "noise": all_epoch_noises
})

shared_source = ColumnDataSource(data={
    "x": X_coord[:, 0],
    "y": X_coord[:, 1],
    "class": y_train,
    "color": ['white'] * len(y_train),
    "marker": [marker[cls] for cls in y_train],
    "size": scaled_sizes_list[0],
    "alpha": scaled_alphas_list[0],
    "sig_in": sig_in[0],
    "fixed_axis": [0] * len(y_train),
    "logits": logits[0],
    "noise": all_epoch_noises[0],
    "selection": [6] * len(y_train),
    "line_color": ['white'] * len(y_train),
    "bar_alpha": [0] * len(y_train)
})

all_barplot = ColumnDataSource(data={
    "noise": region_means_list,
    "sig_in": region_sig_in_list
})

current_barplot = ColumnDataSource(data={
    "noise": region_means_list[0],
    "color": ['white'] * len(region_means_list[0]),
    "sig_in": region_sig_in_list[0]
})

min_y = np.min(region_means_list)
max_y = np.max(region_means_list)

boundary = LSBoundaryVisualizer(shared_source, shared_resource, max_step-1, colors, total_batches, mode='Step', sig_projection=True, barplot_shared_resource=all_barplot, barplot_shared_source=current_barplot)
#projection = LinePlot(shared_source, min_x=np.min(sig_in), max_x=np.max(sig_in))
sigmoid = ProjectionPlot(shared_source, min_x=np.min(sig_in), max_x=np.max(sig_in))
barplot = BarProjectionPlot(current_barplot, shared_source, min_x=np.min(region_sig_in_list), max_x=np.max(region_sig_in_list), min_y=min_y, max_y=max_y)

boundary_layout = column(boundary.get_layout())
sigmoid_layout = column(sigmoid.get_layout())
#projection_layout = column(projection.get_layout())
barplot_layout = column(barplot.get_layout())


layout = row(
    boundary_layout, 
    column(barplot_layout,
           sigmoid_layout,
           #projection_layout
           ), 
    )

curdoc().add_root(layout)

if args.output is not None:
    layout.sizing_mode = "scale_both"
    save(layout)