import h5py
from bokeh.plotting import curdoc, output_file, save
from bokeh.models import ColumnDataSource
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
from visualizer.image_memorymap import ImageSensitivityVisualizer

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
    bpe_scores = [f[f"scores/epoch_{epoch}"]["bpe"][()] for epoch in range(max_epoch)]
    bls_scores = [f[f"scores/epoch_{epoch}"]["bls"][()] for epoch in range(max_epoch)]
    all_epoch_noises = [f[f"scores/epoch_{epoch}"]["noise"][()] for epoch in range(max_epoch)]

    test_acc = [f[f"results/epoch_{epoch}"]["test_acc"][()] for epoch in range(max_epoch)]
    test_nll = [f[f"results/epoch_{epoch}"]["test_nll"][()] for epoch in range(max_epoch)]
    estimated_nll = [f[f"results/epoch_{epoch}"]["estimated_nll"][()] for epoch in range(max_epoch)]


if args.compress:
    sample_size = min(args.n_sample, len(labels))
    sample_indices = np.random.choice(len(labels), sample_size, replace=False)

    sample_bpe = [np.array(epoch_scores)[sample_indices] for epoch_scores in bpe_scores]
    sample_bls = [np.array(epoch_scores)[sample_indices] for epoch_scores in bls_scores]

    sample_labels = labels[sample_indices]
    sample_images = images[sample_indices]

    bpe_scores = sample_bpe
    bls_scores = sample_bls
    labels = sample_labels
    images = sample_images
    

# Convert all images in sorted order
if dataset == 'MNIST':
    image_base64_list = [mnist_to_base64(img) for img in images]
elif dataset == 'CIFAR10':
    image_base64_list = [cifar10_to_base64(img) for img in images]

epoch_orders = [np.argsort(noise) for noise in all_epoch_noises]


shared_resource = ColumnDataSource(data={
    "bpe": bpe_scores,
    "bls": bls_scores,
    #"sensitivities": sentivities,
    "epoch": list(range(max_epoch)),
    #"noises": all_epoch_noises,
    #"order": epoch_orders
})

shared_source = ColumnDataSource(data={
    "img": image_base64_list,
    "label": labels,
    "bpe": bpe_scores[0],
    "bls": bls_scores[0],
    #"sensitivities": sentivities[0],
    "size": [6] * len(labels),
    "alpha": [1.0] * len(labels),
    "color": ['blue'] * len(labels),
    "marker": ['circle'] * len(labels),
    #"noises": all_epoch_noises[0],
    #"order": epoch_orders[0]
})

memorymapvisualizer = ImageSensitivityVisualizer(shared_source, shared_resource, max_epoch)

memory_layout = column(memorymapvisualizer.get_layout(), width=600)

layout = row(memory_layout)

curdoc().add_root(layout)

save(layout)