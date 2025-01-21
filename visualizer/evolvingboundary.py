from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models import Div, Slider, ColumnDataSource, Button, Spacer, InlineStyleSheet, Div
from bokeh.models.widgets import RadioButtonGroup
import numpy as np
from skimage import measure
from bokeh.io import curdoc
import matplotlib.pyplot as plt
import matplotlib

class EvolvingBoundaryVisualizer:
    def __init__(self, shared_source, shared_resource, mod1, epoch_steps, colors, max_epochs=30):
        self.source = shared_source
        self.shared_resource = shared_resource
        self.mod1 = mod1
        self.epoch_steps = epoch_steps

        self.X = np.column_stack([self.source.data[feature] for feature in self.source.data if feature in ['x', 'y']])
        self.y = self.source.data['class']

        self.classes = np.unique(self.y)
        self.message_div = Div(text="", width=400, height=50, styles={"color": "black"})

        self.colors = colors
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

        self.boundary_source = ColumnDataSource(data={"xs": [], "ys": [], "prev_xs": [], "prev_ys": []})

        self.plot.scatter("x", "y", size=8, source=self.source, color="color", marker="marker")
        self.plot.multi_line(xs="xs", ys="ys", source=self.boundary_source, line_width=2, color="black")
        self.plot.multi_line(xs="prev_xs", ys="prev_ys", source=self.boundary_source, line_width=2, color="grey")

        self.epoch = 0
        self.max_epochs = max_epochs

        xx, yy, zz = self.calculate_boundaries()
        self.mod1.update()
        self.update_boundary(xx, yy, zz)

        # Slider for epoch control
        self.epoch_slider = Slider(start=0, end=self.max_epochs, value=0, step=1, title="Epoch")
        self.epoch_slider.on_change('value', self.slider_update)

        self.clear_button = Button(label="Clear Selection", button_type="danger")
        self.clear_button.on_click(self.reset_selection)

        # Colors for tracker buttons (using matplotlib tab10 colors)
        self.tracker_colors = [plt.cm.tab10(i) for i in range(10)]  # Store as RGBA
        self.tracker_colors_hex = [matplotlib.colors.rgb2hex(c) for c in self.tracker_colors]  # Store as hex

        # Creating individual buttons for color selection
        self.tracker_buttons = []
        for i, color in enumerate(self.tracker_colors_hex):
            button = Button(label=f"Color {i + 1}", width=100, button_type="primary")
            button.css_classes = [f"color-button-{i}"]
            button.on_click(lambda color=color: self.apply_tracker_color(color))
            self.tracker_buttons.append(button)

        # Play/Pause button
        self.play_pause_button = Button(label="Play", width=100)
        self.play_pause_button.on_click(self.toggle_play_pause)
        self.running = False  # Tracks whether animation is running

    def toggle_play_pause(self):
        if self.running:
            self.pause_animation()
        else:
            self.start_animation()

    def start_animation(self):
        self.running = True
        self.play_pause_button.label = "Pause"
        self.animate()

    def pause_animation(self):
        self.running = False
        self.play_pause_button.label = "Play"

    def animate(self):
        if self.running and self.epoch < self.max_epochs:
            self.epoch += 1
            self.epoch_slider.value = self.epoch  # Triggers `slider_update`
            curdoc().add_timeout_callback(self.animate, 100)  # 100ms interval
        elif self.epoch >= self.max_epochs:
            self.pause_animation()

    def reset_selection(self):
        new_data = self.source.data.copy()
        for idx in range(len(new_data["color"])):
            new_data["color"][idx] = self.colors[int(new_data["class"][idx])]

        self.source.data = new_data
        self.source.selected.indices = []  # Clear selection
        self.message_div.text = "Selections cleared."

    def apply_tracker_color(self, color):
        selected_indices = self.source.selected.indices
        if not selected_indices:
            self.message_div.text = "No points selected to apply color."
            return

        new_data = self.source.data.copy()
        for idx in selected_indices:
            new_data["color"][idx] = color

        self.source.data = new_data
        self.message_div.text = f"Applied color '{color}' to selected points."

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
            if self.epoch > 1:
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

    def slider_update(self, attr, old, new):
        self.epoch = new
        xx, yy, zz = self.calculate_boundaries()
        self.update_boundary(xx, yy, zz)
        self.message_div.text = f"Epoch {self.epoch}/{self.max_epochs} selected."

    def get_layout(self):
        return column(
            self.plot,
            self.message_div,
            self.play_pause_button,
            row(Spacer(width=50),self.epoch_slider, Spacer(width=50)),
            row(Div(text="Tracker Colors:"), *self.tracker_buttons),
            row(self.clear_button)
        )