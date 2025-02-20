import h5py
from bokeh.plotting import curdoc, output_file, save
from visualizer.evolvingboundary import EvolvingBoundaryVisualizer
from visualizer.evolvingmpe import EvolvingMemoryMapVisualizer
from visualizer.evolvingsensitivity import EvolvingSensitivityVisualizer
from bokeh.models import ColumnDataSource
from bokeh.layouts import column, row
import json
import sys
import argparse
import os
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import matplotlib.cm as cm
from io import BytesIO
import base64
from visualizer.labelnoise import LabelNoisePlot
import os

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

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Launch the Bokeh server with an HDF5 file.")
parser.add_argument("--file", type=str, required=True, help="Path to the HDF5 file")
parser.add_argument("--memory_map", type=bool, default=False, help="if True display memory map too")
parser.add_argument("--output", type=str, required=False, help="If specified filename, while running on python not bokeh serve, the html will be saved under ./output")
args = parser.parse_args()

if args.output is not None:
    os.makedirs('./output', exist_ok=True)
    output_file(filename=f"./output/{args.output}.html", title="Static HTML file", mode="inline")

# Load the HDF5 file
h5_file = args.file

# Check if the file has an .h5 extension
h5_file = args.file
if not h5_file.lower().endswith(".h5"):
    print(f"Error: The input file '{h5_file}' is not an HDF5 (.h5) file.")
    sys.exit(1)

# Check if the file exists
if not os.path.isfile(h5_file):
    print(f"Error: The file '{h5_file}' does not exist.")
    sys.exit(1)

with h5py.File(h5_file, "r") as f:
    all_noise = np.array(f["noise"])  # Load noise values
    images = np.array(f["images"])  # MNIST images
    labels = np.array(f["labels"])  # Corresponding labels
    bpe = np.array(f['bpe'])
    bls = np.array(f['bls'])
    read = f["config"]["config_data"][()]
    config_json = read.decode("utf-8")
    config = json.loads(config_json)
    dataset = config.get("dataset")


# Sort data based on noise
sort_noises, index, labels, bpe, bls = zip(*sorted(zip(all_noise, range(len(all_noise)), labels, bpe, bls), reverse=True))
sort_noises = np.array(sort_noises)
index = np.array(index)
labels = np.array(labels)
bpe = np.array(bpe)
bls = np.array(bls)

# Convert all images in sorted order
if dataset == 'MNIST':
    image_base64_list = [mnist_to_base64(images[i]) for i in index]
elif dataset == 'CIFAR10':
    image_base64_list = [cifar10_to_base64(images[i]) for i in index]

# Prepare Data for Bokeh
if args.memory_map:
    source = ColumnDataSource(data={
        "x": list(range(len(sort_noises))),
        "y": sort_noises,
        "label": labels.astype(str),  # Convert labels to string for tooltip
        "img": image_base64_list,  # Add base64 images
        "color": ['grey'] * len(sort_noises),
        "bpe": bpe,
        "bls": bls,
        "marker": ['square'] * len(sort_noises),
        "alpha": [1.0] * len(sort_noises),
        "size": [6] * len(sort_noises)
    })
else:
    source = ColumnDataSource(data={
        "x": list(range(len(sort_noises))),
        "y": sort_noises,
        "label": labels.astype(str),  # Convert labels to string for tooltip
        "img": image_base64_list,  # Add base64 images
        "color": ['grey'] * len(sort_noises)
    })
    
labelnoise = LabelNoisePlot(source, args.memory_map)

labelnoise_layout = column(labelnoise.get_layout(), width=800, height=600)

layout = row(labelnoise_layout)

curdoc().add_root(layout)
save(layout)