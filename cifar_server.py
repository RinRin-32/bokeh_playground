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
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import matplotlib.cm as cm
from io import BytesIO
import base64
from visualizer.labelnoise import LabelNoisePlot
import sys
import torch

sys.path.append("../memory-perturbation")

from lib.datasets import get_dataset

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Launch the Bokeh server with an HDF5 file.")
parser.add_argument("--file", type=str, required=True, help="Path to the HDF5 file")
args = parser.parse_args()


data = np.load(args.file)

ds_train, ds_test, transform_train = get_dataset('CIFAR10', return_transform=True)

all_noise = data["label_noise_all"]  # Load noise values
all_noise = [np.linalg.norm(x,2) for x in all_noise]
#load image here directly from Cifar 10
n_samples = len(ds_train)
index=list(range(n_samples))
images = torch.stack([ds_train[i][0].squeeze() for i in index]).numpy()

labels = np.array([CIFAR10_CLASSES[int(label)] for label in data["labels_all"]]) # Corresponding labels

# Sort data based on noise
sort_noises, index, labels = zip(*sorted(zip(all_noise, range(len(all_noise)), labels), reverse=True))
sort_noises = np.array(sort_noises)
index = np.array(index)
labels = np.array(labels)

def image_to_base64(image_array):
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

# Convert all images in sorted order
image_base64_list = [image_to_base64(images[i]) for i in index]

# Prepare Data for Bokeh
source = ColumnDataSource(data=dict(
    x=list(range(len(sort_noises))),
    y=sort_noises,
    label=labels.astype(str),  # Convert labels to string for tooltip
    img=image_base64_list  # Add base64 images
))

labelnoise = LabelNoisePlot(source)

labelnoise_layout = column(labelnoise.get_layout(), width=800, height=600)

layout = row(labelnoise_layout)

curdoc().add_root(layout)