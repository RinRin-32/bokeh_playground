from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models import Div, Button, ColumnDataSource
import numpy as np
from skimage import measure

from bokeh.io import curdoc

class EvolvingBoundaryVisualizer:
    def __init__(self, shared_source, shared_resource, mod1, epoch_steps, max_epochs=30):
        self.source = shared_source
        self.shared_resource = shared_resource
        self.mod1 = mod1
        self.epoch_steps = epoch_steps

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
            y_range=(y_min, y_max),
            tools="tap,box_select,box_zoom,reset,pan",
            active_drag="box_select"
        )

        self.boundary_source = ColumnDataSource(data=dict(xs=[], ys=[], prev_xs=[], prev_ys=[]))

        self.plot.scatter("x", "y", size=8, source=self.source, color="color", marker="marker")
        self.plot.multi_line(xs="xs", ys="ys", source=self.boundary_source, line_width=2, color="black")
        self.plot.multi_line(xs="prev_xs", ys="prev_ys", source=self.boundary_source, line_width=2, color="grey")

        self.epoch = 0
        self.max_epochs = max_epochs
        self.running = False

        xx, yy, zz = self.calculate_boundaries()
        self.mod1.update()
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
        unique_classes = np.unique(self.y)
        if len(unique_classes) < 2:
            self.message_div.text = "Error: At least two classes are required to fit the model."
            return None, None, None
        else:
            self.message_div.text = ""

            shared_data = self.shared_resource.data
            if self.epoch in shared_data["epoch"]:
                epoch_index = shared_data["epoch"].index(self.epoch)
                xx = shared_data["xx"][epoch_index]
                yy = shared_data["yy"][epoch_index]
                zz = shared_data["Z"][epoch_index]
            if self.epoch % self.epoch_steps == 0:
                new_data = self.source.data.copy()
                shared_data = self.shared_resource.data
                if self.epoch in shared_data["epoch"]:
                    epoch_index = shared_data["epoch"].index(self.epoch)
                    bls = shared_data["bls"][epoch_index]
                    bpe = shared_data["bpe"][epoch_index]
                    sensitivity = shared_data["sensitivities"][epoch_index]
                    softmax_deviations = shared_data["softmax_deviations"][epoch_index]
                    new_data["bls"] = bls
                    new_data["bpe"] = bpe
                    new_data["sensitivities"] = sensitivity
                    new_data["softmax_deviations"] = softmax_deviations
                    self.source.data = new_data
                    self.mod1.update()
            self.epoch += 1

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
        self.boundary_source.data = {"xs": [], "ys": [], "prev_xs": [], "prev_ys": []}

        xx, yy, zz = self.calculate_boundaries()
        self.mod1.update()
        self.update_boundary(xx, yy, zz)
        self.message_div.text = "Reset complete. Ready to start."

    def get_layout(self):
        return column(
            self.plot,
            self.message_div,
            row(self.play_button, self.pause_button, self.reset_button)
        )

    def animate(self):
        if self.epoch <= self.max_epochs:
            xx, yy, zz = self.calculate_boundaries()
            self.update_boundary(xx, yy, zz)
            self.message_div.text = f"Epoch {self.epoch-1}/{self.max_epochs} completed."
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