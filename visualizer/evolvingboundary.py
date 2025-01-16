from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models import Div, Button, ColumnDataSource, GlyphRenderer, Image
import numpy as np
from torch import nn
import torch
from torch.utils.data import TensorDataset
from skimage import measure  # To extract contour lines
import sys

from bokeh.io import curdoc

sys.path.append("../memory-perturbation")

from lib.utils import train_model, predict_test

from lib.datasets import get_dataset

from lib.utils import get_quick_loader

from torch.utils.data import DataLoader
from lib.models import get_model
from ivon import IVON as IBLR

class EvolvingBoundaryVisualizer:
    def __init__(self, shared_source):
        self.source = shared_source

        self.X = np.column_stack([self.source.data[feature] for feature in self.source.data if feature in ['x', 'y']])
        self.y = self.source.data['class']

        self.classes = np.unique(self.y) 
        self.message_div = Div(text="", width=400, height=50, styles={"color": "black"})

        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1

        self.plot = figure(
            title="Evolving Boundary Visualization", 
            width=600, height=600, 
            x_range=(x_min, x_max), 
            y_range=(y_min, y_max)
        )

        self.boundary_source = ColumnDataSource(data=dict(xs=[], ys=[], prev_xs=[], prev_ys=[]))

        self.plot.scatter("x", "y", size=8, source=self.source, color="color", marker="marker")
        self.plot.multi_line(xs="xs", ys="ys", source=self.boundary_source, line_width=2, color="black")
        self.plot.multi_line(xs="prev_xs", ys="prev_ys", source=self.boundary_source, line_width=2, color="grey")

        self.model = get_model('small_mlp', 2, 2, 'cuda', 1)
        self.optim = IBLR(self.model.parameters(), lr=2, mc_samples=4, ess=800, weight_decay=1e-3,
                          beta1=0.9, beta2=0.99999, hess_init=0.9)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=30)
        self.criterion = nn.CrossEntropyLoss().to('cuda')

        self.ds_train = TensorDataset(torch.tensor(self.X, dtype=torch.float32), torch.tensor(self.y, dtype=torch.long))
        self.trainloader = get_quick_loader(DataLoader(self.ds_train, batch_size=256, shuffle=False), device='cuda')

        self.epoch = 0
        self.max_epochs = 30
        self.running = False

        xx, yy, zz = self.calculate_boundaries()
        self.update_boundary(xx, yy, zz)

        # Buttons for control
        self.play_button = Button(label="Play", width=100)
        self.pause_button = Button(label="Pause", width=100)
        self.reset_button = Button(label="Reset", width=100)

        self.play_button.on_click(self.start_animation)
        self.pause_button.on_click(self.pause_animation)
        self.reset_button.on_click(self.reset)

        self.layout = column(
            self.plot,
            self.message_div,
            row(self.play_button, self.pause_button, self.reset_button)
        )

        # Store callback id for tracking
        self.animation_callback_id = None

    def calculate_boundaries(self):
        print("Calculating boundaries...")
        unique_classes = np.unique(self.y)
        if len(unique_classes) < 2:
            self.message_div.text = "Error: At least two classes are required to fit the model."
            return None, None, None
        else:
            self.message_div.text = ""
            self.epoch += 1
            self.model, self.optim = train_model(self.model, self.criterion, self.optim, self.scheduler, self.trainloader, 1, 799, None, 'cuda', return_optim=True)
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

            # Update the ColumnDataSource with both the current and previous boundaries
            current_data = self.boundary_source.data
            if self.epoch>1:
                prev_xs = current_data["xs"]
                prev_ys = current_data["ys"]
            
                self.boundary_source.data = {
                    "xs": xs,
                    "ys": ys,
                    "prev_xs": prev_xs,
                    "prev_ys": prev_ys
                }
            else:
                self.boundary_source.data = {
                    "xs": xs,
                    "ys": ys,
                }
        else:
            self.boundary_source.data = {"xs": [], "ys": [], "prev_xs": [], "prev_ys": []}

    def update(self, attr, old, new):
        xx, yy, zz = self.calculate_boundaries()
        self.update_boundary(xx, yy, zz)

    def reset(self, event):
        self.pause_animation()
        self.epoch = 0
        self.model = get_model('small_mlp', 2, 2, 'cuda', 1)
        self.optim = IBLR(self.model.parameters(), lr=2, mc_samples=4, ess=800, weight_decay=1e-3,
                          beta1=0.9, beta2=0.99999, hess_init=0.9)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=30)
        self.criterion = nn.CrossEntropyLoss().to('cuda')
        self.boundary_source.data = {"xs": [], "ys": [], "prev_xs": [], "prev_ys": []}

        xx, yy, zz = self.calculate_boundaries()
        self.update_boundary(xx, yy, zz)
        self.message_div.text = "Reset complete. Ready to start."

    def get_layout(self):
        return column(
            self.plot,
            self.message_div,
            row(self.play_button, self.pause_button, self.reset_button)
        )

    def animate(self):
        if self.epoch < self.max_epochs:
            xx, yy, zz = self.calculate_boundaries()
            self.update_boundary(xx, yy, zz)
            self.message_div.text = f"Epoch {self.epoch}/{self.max_epochs} completed."
        else:
            self.pause_animation()
            self.message_div.text = "Training complete."

    def start_animation(self):
        if not self.running:
            self.running = True
            self.animation_callback_id = curdoc().add_periodic_callback(self.animate, 500)

    def pause_animation(self):
        if self.animation_callback_id is not None:
            curdoc().remove_periodic_callback(self.animation_callback_id)
            self.animation_callback_id = None
        self.running = False