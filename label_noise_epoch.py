from visualizer.evolving_ls import EvolvingLabelNoisePlot
from visualizer.test_nll import TestNLLAnimation
import h5py
from bokeh.plotting import curdoc, output_file, save
from bokeh.models import ColumnDataSource, Spacer
from bokeh.layouts import column, row
import json
import sys
import argparse
import os
import numpy as np
from PIL import Image
import matplotlib.cm as cm
from io import BytesIO
import base64
import os
import matplotlib.pyplot as plt

def mnist_to_base64(image_array):
    image_array = np.squeeze(image_array, axis=0)  # Remove channel dim -> (28, 28)
    
    # Normalize to range [0, 1] for colormap
    image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
    
    # Apply colormap
    colored_image = cm.gray(image_array)  # Get RGBA values
    
    # Convert to uint8 and remove alpha channel
    img = Image.fromarray((colored_image[..., :3] * 255).astype(np.uint8))  # Use RGB only
    
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def cifar10_to_base64(image_array):
    # Ensure image is (H, W, 3)
    if image_array.shape[0] == 3:  # (3, 32, 32) → (32, 32, 3)
        image_array = image_array.transpose(1, 2, 0)
    
    # Normalize to [0, 255] and convert to uint8
    image_array = ((image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255).astype(np.uint8)
    
    # Convert NumPy array to PIL Image
    img = Image.fromarray(image_array)  # Now it correctly handles RGB
    
    # Encode to base64
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def generate_noise_barchart(noise_values, width=150, height=100, dpi=100):
    # Create a bar chart from the noise values
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=dpi)
    ax.bar(range(len(noise_values)), noise_values, color='gray')
    # Remove ticks and labels for a clean image
    ax.set_xticks(range(len(noise_values)))
    ax.set_xticklabels([f"{i}" for i in range(len(noise_values))], fontsize=8)

    ax.set_yticks([])
    plt.tight_layout()

    # Save the figure to a bytes buffer and encode as base64
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    #plt.savefig('./test.png', format="png", bbox_inches='tight')
    #raise RuntimeError('printed')
    buf.flush()
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

parser = argparse.ArgumentParser(description="Launch the Bokeh server displaying Label Smoothing plot with an HDF5 file.")
parser.add_argument("--file", type=str, required=True, help="Path to the HDF5 file")
parser.add_argument("--compress", action="store_true", help="Enable random sampling of images")
parser.add_argument("--no-compress", dest="compress", action="store_false", help="Disable random sampling of images")
parser.add_argument("--n_sample", type=int, default=1000, help="Number of images selected for plot if compressing, 1000 by default")
parser.add_argument("--output", type=str, required=False, help="If specified filename, while running on python not bokeh serve, the html will be saved under ./output")
args = parser.parse_args()

if args.output is not None:
    os.makedirs('./output', exist_ok=True)
    output_file(filename=f"./output/{args.output}.html", title="Static HTML file", mode="inline")

h5_file = args.file

if not h5_file.lower().endswith(".h5"):
    print(f"Error: The input file '{h5_file}' is not an HDF5 (.h5) file.")
    sys.exit(1)

# Check if the file exists
if not os.path.isfile(h5_file):
    print(f"Error: The file '{h5_file}' does not exist.")
    sys.exit(1)

with h5py.File(h5_file, "r") as f:
    read = f["config"]["config_data"][()]
    config_json = read.decode("utf-8")
    config = json.loads(config_json)
    dataset = config.get("dataset")
    max_epoch = config.get("max_epochs")

    images = np.array(f["images"])
    labels = np.array(f["labels"])

    sentivities = [f[f"scores/epoch_{epoch}"]["sensitivities"][()] for epoch in range(max_epoch)]
    all_epoch_noises = [f[f"scores/epoch_{epoch}"]["noise"][()] for epoch in range(max_epoch)]

    all_induced_noises = [f[f"scores/epoch_{epoch}"]["all_noise"][()] for epoch in range(max_epoch)]

    test_acc = [f[f"results/epoch_{epoch}"]["test_acc"][()] for epoch in range(max_epoch)]
    test_nll = [float(f[f"results/epoch_{epoch}"]["test_nll"][()].item()) for epoch in range(max_epoch)]    
    estimated_nll = [f[f"results/epoch_{epoch}"]["estimated_nll"][()] for epoch in range(max_epoch)]


if args.compress:
    sample_size = min(args.n_sample, len(labels))
    sample_indices = np.random.choice(len(labels), sample_size, replace=False)

    sample_noise = [np.array(epoch_scores)[sample_indices] for epoch_scores in all_epoch_noises]

    sample_induced_noise = [np.array(noise)[sample_indices] for noise in all_induced_noises]

    sample_labels = labels[sample_indices]
    sample_images = images[sample_indices]

    all_epoch_noises = sample_noise
    labels = sample_labels
    images = sample_images
    all_induced_noises = sample_induced_noise
    

# Convert all images in sorted order
if dataset == 'MNIST':
    image_base64_list = [mnist_to_base64(img) for img in images]
elif dataset == 'CIFAR10':
    image_base64_list = [cifar10_to_base64(img) for img in images]

all_epoch_indices = [np.argsort(noises)[::-1] for noises in all_epoch_noises]

relative_positioning = []
for indices in all_epoch_indices:
    # Create a new array of the same size
    relative_position = np.zeros_like(indices)
    
    # Fill relative_position such that for each sorted position, we store the original index
    for sorted_index, original_index in enumerate(indices):
        relative_position[original_index] = sorted_index
    
    # Append the relative_position for the current epoch
    relative_positioning.append(relative_position)

# Extract min and max across all epochs
y_min = min(np.min(noises) for noises in all_epoch_noises)
y_max = max(np.max(noises) for noises in all_epoch_noises)

# Store them as a list
y_range = [y_min, y_max]

# Normalize each epoch's noise independently
normalized_induced_noises = []

for epoch_noises in all_induced_noises:  # Each epoch
    epoch_noises = np.array(epoch_noises)  # Shape: (num_datapoints, num_classes)

    # Convert noise values to absolute (as higher absolute noise means higher confidence)
    abs_noises = np.abs(epoch_noises)

    # Normalize so each row (datapoint) sums to 1
    row_sums = np.sum(abs_noises, axis=1, keepdims=True)  # Shape: (num_datapoints, 1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero

    normalized_noises = abs_noises / row_sums  # Shape: (num_datapoints, num_classes)

    normalized_induced_noises.append(normalized_noises.tolist())  # Store as list


noise_barcharts = []
for epoch in range(len(normalized_induced_noises)):
    epoch_chart = []
    for images in range(len(normalized_induced_noises[epoch])):
        epoch_chart.append(generate_noise_barchart(normalized_induced_noises[epoch][images]))
    noise_barcharts.append(epoch_chart)

normalized_induced_noises = np.array(normalized_induced_noises).reshape(len(normalized_induced_noises), -1, 10)


shared_resource = ColumnDataSource(data={
    "y": all_epoch_noises,
    "test_nll": test_nll,
    "estimated_nll": estimated_nll,
    "epoch": list(range(max_epoch)),
    "x": relative_positioning,
    "noise_chart": noise_barcharts,
#    "induced_noise": normalized_induced_noises
})

#get all the index here somehow to reduce computation and checks required done in the jscallbacks
shared_source = ColumnDataSource(data={
    "img": image_base64_list,
    "label": labels.astype(str),
    "size": [6] * len(labels),
    "alpha": [1.0] * len(labels),
    "color": ['blue'] * len(labels),
    "marker": ['circle'] * len(labels),
    "y": all_epoch_noises[0],
    "x": relative_positioning[0],
    "noise_chart": noise_barcharts[0],
#    "induced_noise": normalized_induced_noises[0]
})

max_epoch-=1

evolving_ls = EvolvingLabelNoisePlot(shared_source, dataset, y_range, len(all_epoch_noises[0]))
nll_plot = TestNLLAnimation(shared_source, shared_resource, max_epoch)

ls_layout = column(evolving_ls.get_layout(), sizing_mode="stretch_width")
nll_layout = column(nll_plot.get_layout(), sizing_mode="stretch_height")

layout = column(row(ls_layout), row(nll_layout), sizing_mode="stretch_both")

curdoc().add_root(layout)

if args.output is not None:
    layout.sizing_mode = "stretch_both" 
    save(layout)



#Notes:


'''
Likely the way to make the noise_chart changes dynamically in evolvingnoiseplot is to pre sort the whole thing, store the index in a list,

this way so that the callback doesn't have to do all the loop and only display according to the datasource

maybe consider removing the images for the noise plot, just to reduce the file size
'''