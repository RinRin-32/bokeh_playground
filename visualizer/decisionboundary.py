from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models import Div, Button, ColumnDataSource, GlyphRenderer, Image
import numpy as np
from torch import nn
import torch
from torch.utils.data import TensorDataset
from skimage import measure  # To extract contour lines
import sys

sys.path.append("../memory-perturbation")

from lib.utils import train_model

from lib.datasets import get_dataset

from lib.utils import get_quick_loader

from torch.utils.data import DataLoader
from lib.models import get_model
from ivon import IVON as IBLR

class DecisionBoundaryVisualizer:
    def __init__(self, shared_source):
        self.source = shared_source

        self.X = np.column_stack([self.source.data[feature] for feature in self.source.data if feature not in ['id', 'class', 'color', 'marker', 'estimated_deviation', 'true_deviation', 'bpe', 'bls']])
        self.y = self.source.data['class']

        self.classes = np.unique(self.y) 
        self.message_div = Div(text="", width=400, height=50, styles={"color": "red"})

        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1

        self.plot = figure(
            title="Interactive 2D Classification Visualization", 
            width=600, height=600, 
            tools="tap,box_select,box_zoom,reset",
            active_drag="box_select",
            x_range=(x_min, x_max), 
            y_range=(y_min, y_max)
        )

        self.boundary_source = ColumnDataSource(data=dict(xs=[], ys=[]))

        self.plot.scatter("x", "y", size=8, source=self.source, color="color", marker="marker")
        self.plot.multi_line(xs="xs", ys="ys", source=self.boundary_source, line_width=2, color="black")

        xx, yy, zz = self.calculate_boundaries(self.X, self.y)
        self.update_boundary(xx, yy, zz)

        self.source.on_change('data', self.update)

    def calculate_boundaries(self, X, y):
        print("Calculating boundaries...")
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            self.message_div.text = "Error: At least two classes are required to fit the model."
            return None, None, None
        else:

            input_size = 2
            nc = 2
            self.model = get_model('small_mlp', nc, input_size, 'cuda', 1)

            optim = IBLR(self.model.parameters(), lr=1e-2, mc_samples=1, ess=len(X), weight_decay=60/len(X),
                                beta1=0.9, beta2=0.99999, hess_init=0.1)

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=20, eta_min=1e-4)

            self.message_div.text = ""
            criterion = nn.CrossEntropyLoss(reduction='mean').to('cuda')
            ds_train, ds_test, transform_train = get_dataset('MOON', return_transform=True, noise=0.2)
            trainloader = get_quick_loader(DataLoader(ds_train, batch_size=1), device='cuda') # training
            self.model, _ = train_model(self.model, criterion, optim, scheduler, trainloader, 20, 799, 60, 'cuda', False)
            self.model.eval()
            
            x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
            y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), 
                                 np.arange(y_min, y_max, 0.01))
            
            grid = np.c_[xx.ravel(), yy.ravel()]
            grid = torch.tensor(grid, dtype=torch.float32).to('cuda')
            
            with torch.no_grad():
                logits = self.model(grid)
            
            probabilities = torch.softmax(logits, dim=1)
            zz = torch.argmax(probabilities, dim=1)
            zz = zz.cpu().numpy().reshape(xx.shape)
            print(zz)
            
            return xx, yy, zz

    def extract_boundary_lines(self, xx, yy, zz):
        contours = measure.find_contours(zz, level=0.5)  # Assuming boundary at 0.5 probability
        xs, ys = [], []
        for contour in contours:
            xs.append(xx[0, 0] + contour[:, 1] * (xx[0, -1] - xx[0, 0]) / zz.shape[1])
            ys.append(yy[0, 0] + contour[:, 0] * (yy[-1, 0] - yy[0, 0]) / zz.shape[0])
        return xs, ys

    def update_boundary(self, xx, yy, zz):
        if xx is not None and yy is not None and zz is not None:
            xs, ys = self.extract_boundary_lines(xx, yy, zz)
            self.boundary_source.data = {"xs": xs, "ys": ys}
        else:
            self.boundary_source.data = {"xs": [], "ys": []}

    def update(self, attr, old, new):
        mask = np.array(self.source.data["color"]) != "grey"
        x_new = np.array(self.source.data["x"])[mask]
        y_new = np.array(self.source.data["y"])[mask]
        
        X_new = np.vstack((x_new, y_new)).T
        y_new = np.array(self.source.data["class"])[mask].flatten()

        xx, yy, zz = self.calculate_boundaries(X_new, y_new)
        self.update_boundary(xx, yy, zz)

    def get_layout(self):
        return column(self.plot, self.message_div)