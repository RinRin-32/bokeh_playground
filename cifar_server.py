from bokeh.plotting import curdoc, output_file, save
from bokeh.models import ColumnDataSource
from bokeh.layouts import column, row
import argparse
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from visualizer.labelnoise import LabelNoisePlot
import torch
import sys
import os

sys.path.append("../memory-perturbation")

from lib.datasets import get_dataset

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

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Launch a Bokeh server with an npz file, this plots label smoothing on CIFAR10.")
parser.add_argument("--file", type=str, required=True, help="Path to the npz file")
parser.add_argument("--compress", action="store_true", help="Enable random sampling of images")
parser.add_argument("--no-compress", dest="compress", action="store_false", help="Disable random sampling of images")
parser.add_argument("--n_sample", type=int, default=1000, help="Number of images selected for plot if compressing, 1000 by default")
parser.add_argument("--output", type=str, required=False, help="If specified filename, while running on python not bokeh serve, the html will be saved under ./output")
args = parser.parse_args()

if args.output is not None:
    os.makedirs('./output', exist_ok=True)
    output_file(filename=f"./output/{args.output}.html", title="Static HTML file", mode="inline")

if not args.file.lower().endswith(".npz"):
    print(f"Error: The input file '{args.file}' is not an .npz file.")
    sys.exit(1)


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

if args.compress:
    # Set sample size
    sample_size = min(args.n_sample, len(sort_noises))  # Adjust based on visualization needs

    # Randomly sample from the sorted indices (to keep sorting intact)
    sample_indices = np.random.choice(len(sort_noises), sample_size, replace=False)

    # Extract sampled data while keeping indexing consistent
    sample_noises = sort_noises[sample_indices]
    sample_cifar_indices = index[sample_indices]  # Original CIFAR-10 indices
    sample_labels = labels[sample_indices]
    sample_images = images[sample_cifar_indices]  # Extract corresponding CIFAR images

    # Sort sampled data by noise values (descending order for visualization)
    sorted_data = sorted(zip(sample_noises, sample_cifar_indices, sample_labels, sample_images), 
                        key=lambda x: x[0], reverse=True)

    # Unpack sorted data
    sorted_noises, sorted_cifar_indices, sorted_labels, sorted_images = zip(*sorted_data)

    sort_noises = np.array(sorted_noises)
    labels = np.array(sorted_labels)


    image_base64_list = [image_to_base64(img) for img in sorted_images]
    
else:
    image_base64_list = [image_to_base64(images[i]) for i in index]

# Prepare Data for Bokeh
source = ColumnDataSource(data=dict(
    x=list(range(len(sort_noises))),
    y=sort_noises,
    label=labels.astype(str),  # Convert labels to string for tooltip
    img=image_base64_list,  # Add base64 images
    color= ['grey'] * len(sorted_noises)
))

labelnoise = LabelNoisePlot(source, 'CIFAR-10')

labelnoise_layout = column(labelnoise.get_layout(), width=800, height=600)

layout = row(labelnoise_layout)

curdoc().add_root(layout)

if args.output is not None:
    save(layout)