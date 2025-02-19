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

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Launch the Bokeh server with an HDF5 file.")
parser.add_argument("--file", type=str, required=True, help="Path to the HDF5 file")
args = parser.parse_args()

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

# Sort data based on noise
sort_noises, index, labels = zip(*sorted(zip(all_noise, range(len(all_noise)), labels), reverse=True))
sort_noises = np.array(sort_noises)
index = np.array(index)
labels = np.array(labels)

def image_to_base64(image_array):
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

'''
#for cifar10 (need to make this param later)
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
'''

# Convert all images in sorted order
image_base64_list = [image_to_base64(images[i]) for i in index]

# Prepare Data for Bokeh
source = ColumnDataSource(data=dict(
    x=list(range(len(sort_noises))),
    y=sort_noises,
    label=labels.astype(str),  # Convert labels to string for tooltip
    img=image_base64_list,  # Add base64 images
    color= ['grey'] * len(sort_noises)
))

labelnoise = LabelNoisePlot(source)

labelnoise_layout = column(labelnoise.get_layout(), width=800, height=600)

layout = row(labelnoise_layout)

curdoc().add_root(layout)