from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.models import Div, ColumnDataSource, Button
import numpy as np
from torch import nn
import torch
from torch.utils.data import TensorDataset
from skimage import measure
import sys

sys.path.append("../memory-perturbation")

from lib.utils import train_model

from lib.utils import get_quick_loader

from torch.utils.data import DataLoader
from lib.models import get_model
from ivon import IVON as IBLR

class DecisionBoundaryVisualizer:
    def __init__(self, shared_source, config, colors):

        self.input_size = config.get("input_size")
        self.nc = config.get("nc")
        self.model_name = config.get("model")
        self.device = config.get("device")
        self.optimizer = config.get("optimizer")
        self.optim_param = config.get("optimizer_params")
        self.max_epochs = config.get("max_epochs")
        self.loss_criterion = config.get("loss_criterion")
        self.n_retrain = config.get("n_retrain")
        self.source = shared_source

        self.X = np.column_stack([self.source.data[feature] for feature in self.source.data if feature in ['x', 'y']])
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

        self.plot.scatter("x", "y", source=self.source, color="color", marker="marker", alpha="alpha", size="size", line_color="black")
        self.plot.multi_line(xs="xs", ys="ys", source=self.boundary_source, line_width=2, color="black")

        xx, yy, zz = self.calculate_boundaries(self.X, self.y)
        self.update_boundary(xx, yy, zz)

        # Setup selection callback
        self.source.selected.on_change('indices', self.update_selection)

        # Create buttons for confirmation and reset
        self.confirm_button = Button(label="Confirm Selection", button_type="success")
        self.confirm_button.on_click(self.confirm_selection)

        self.reset_button = Button(label="Reset Selection", button_type="danger")
        self.reset_button.on_click(self.reset_selection)

        self.inverse_button = Button(label="Invert Selection", button_type="primary")
        self.inverse_button.on_click(self.invert_selection)

        self.message_div = Div(text="", width=400, height=25)
        self.colors = colors
        self.ind = []

    def update_selection(self, attr, old, new):
        selected_indices = self.source.selected.indices
        new_data = self.source.data.copy()

        for idx in range(len(new_data['color'])):
            if idx in selected_indices:
                if new_data["color"][idx] != "red":
                    new_data["color"][idx] = "red"
                #else:
                #    new_data["color"][idx] = self.colors[int(new_data["class"][idx])]

        self.source.data = new_data
        self.ind.extend(selected_indices)

    def confirm_selection(self):
        new_data = self.source.data.copy()

        for idx in self.ind:
            # Confirm selected points by setting color to grey
            if new_data["color"][idx] == "red":
                new_data["color"][idx] = "grey"
                new_data["alpha"][idx] = "0.1"

        self.source.data = new_data
        self.source.selected.indices = []  # Clear selection
        self.message_div.text = "Selection confirmed."
        self.update(None, None, None)

    def reset_selection(self):
        new_data = self.source.data.copy()

        for idx in range(len(new_data["color"])):
            new_data["color"][idx] = self.colors[int(new_data["class"][idx])]
            new_data["alpha"][idx] = 1

        self.source.data = new_data
        self.source.selected.indices = []  # Clear selection
        self.message_div.text = "Selections reset."
        self.update(None, None, None)

    def invert_selection(self):
        new_data = self.source.data.copy()
        count = 0

        for idx in range(len(new_data["color"])):
            if new_data["color"][idx] != 'grey':
                new_data["color"][idx] = 'grey'
                new_data["alpha"][idx] = "0.1"
            elif new_data["color"][idx] == 'red':
                pass
            else:
                if new_data["color"][idx] != self.colors[int(new_data["class"][idx])]:
                    new_data["color"][idx] = self.colors[int(new_data["class"][idx])]
                    new_data["alpha"][idx] = 1
                else:
                    count+=1
        if count < len(new_data["color"]):
            print("inverted")
            self.source.data = new_data
            self.update(None, None, None)
        else:
            self.reset_selection
        self.source.selected.indices = []

    def calculate_boundaries(self, X, y):
        print("Calculating boundaries...")
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            self.message_div.text = "Error: At least two classes are required to fit the model."
            return None, None, None
        else:

            self.model = get_model(self.model_name, self.nc, self.input_size, self.device, 1)
            optim = IBLR(self.model.parameters(), lr=self.optim_param['lr'], mc_samples=4, ess=self.n_retrain, weight_decay=1e-3,
                                beta1=0.9, beta2=0.99999, hess_init=self.optim_param['hess_init'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=self.max_epochs)

            self.message_div.text = ""
            criterion = nn.CrossEntropyLoss().to(self.device)
            ds_train = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
            trainloader = get_quick_loader(DataLoader(ds_train, batch_size=256, shuffle=False), device=self.device) # training
            self.model, _ = train_model(self.model, criterion, optim, scheduler, trainloader, self.max_epochs, self.n_retrain, None, self.device)
            self.model.eval()
            
            x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
            y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), 
                                 np.arange(y_min, y_max, 0.01))
            
            grid = np.c_[xx.ravel(), yy.ravel()]
            grid = torch.tensor(grid, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                logits = self.model(grid)
            
            probabilities = torch.softmax(logits, dim=1)
            zz = torch.argmax(probabilities, dim=1)
            zz = zz.cpu().numpy().reshape(xx.shape)
            
            return xx, yy, zz

    def extract_boundary_lines(self, xx, yy, zz):
        contours = measure.find_contours(zz, level=0.5)
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
        return column(self.plot, self.message_div, self.confirm_button, self.reset_button, self.inverse_button)