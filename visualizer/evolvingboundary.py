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
    def __init__(self, shared_source, shared_resource, mod1, steps, colors, max_steps=30):
        self.source = shared_source
        self.shared_resource = shared_resource
        self.mod1 = mod1
        self.steps = steps

        self.X = np.column_stack([self.source.data[feature] for feature in self.source.data if feature in ['x', 'y']])
        self.y = self.source.data['class']

        self.classes = np.unique(self.y)
        self.message_div = Div(text="", width=400, height=50, styles={"color": "black"})

        self.colors = colors
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1

        self.plot = figure(
            title="Evolving Boundary Visualization over 30 Epoch",
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

        self.step = 0
        self.max_steps = max_steps

        xx, yy, zz = self.calculate_boundaries()
        self.mod1.update()
        self.update_boundary(xx, yy, zz)

        # Slider for step control
        self.step_slider = Slider(start=0, end=self.max_steps, value=0, step=1, title=f"Epoch: {self.step // 4} Step")
        self.step_slider.on_change('value', self.slider_update)

        self.clear_button = Button(label="Clear Selection", button_type="danger")
        self.clear_button.on_click(self.reset_selection)

        # Colors for tracker buttons (using matplotlib tab10 colors)
        self.tracker_colors = [plt.cm.tab10(i+3) for i in range(6)]  # Store as RGBA
        self.tracker_colors_hex = [matplotlib.colors.rgb2hex(c) for c in self.tracker_colors]  # Store as hex

        # Creating individual buttons for color selection
        self.tracker_buttons = []
        tracker_buttons = []
        for i, color in enumerate(self.tracker_colors_hex):
            # Dynamically create the style for each button
            style = InlineStyleSheet(css=f"""
            :host(.color-button-{i}) {{
                background-color: {color};  /* Solid background color */
                font-weight: bold;
                border: none;
                border-radius: 5px;
                padding: 5px 10px;
                text-align: center;
                cursor: pointer;
            }}
            :host(.color-button-{i}:hover) {{
                opacity: 0.8;
            }}
            """)

            # Create button and assign a unique class
            button = Button(label=f"Group {i + 1}", width=100, stylesheets=[style], css_classes=[f'color-button-{i}'])
            button.on_click(lambda color=color: self.apply_tracker_color(color))
            self.tracker_buttons.append(button)

        # Play/Pause button
        self.play_pause_button = Button(label="Play", width=100)
        self.play_pause_button.on_click(self.toggle_play_pause)
        self.running = False  # Tracks whether animation is running

        # Proceed 1 Step button
        self.proceed_button = Button(label="Next", button_type="success")
        self.proceed_button.on_click(self.next)

        # Adding back button
        self.backtrack_button = Button(label="Previous", button_type="warning")
        self.backtrack_button.on_click(self.back)

        # Forward 1 Epoch
        self.forward_epoch_button = Button(label="Forward 1 Epoch", button_type="success")
        self.forward_epoch_button.on_click(self.forward_epoch)

        # Backward 1 Epoch
        self.backward_epoch_button = Button(label="Backward 1 Epoch", button_type="warning")
        self.backward_epoch_button.on_click(self.backward_epoch)

        self.reset_button = Button(label="Reset", button_type="danger")
        self.reset_button.on_click(self.reset)

    def reset(self):
        self.step = 0
        self.step_slider.value = self.step

    # Methods for the buttons
    def forward_epoch(self):
        if self.step + 4 <= self.max_steps:
            self.step += 4
            self.step_slider.value = self.step
        else:
            self.message_div.text = "You cannot go beyond the maximum step."

    def backward_epoch(self):
        if self.step - 4 >= 0:
            self.step -= 4
            self.step_slider.value = self.step
        else:
            self.message_div.text = "You are already at the first step."

    def next(self):
        if self.step < self.max_steps:
            self.step += 1
            self.step_slider.value = self.step
        else:
            self.message_div.text = "You are already at the last step"

    def back(self):
        if self.step > 0:
            self.step -= 1
            self.step_slider.value = self.step
        else:
            self.message_div.text = "You are already at the first step"

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
        if self.running and self.step < self.max_steps:
            self.step += 1
            self.step_slider.value = self.step  # Triggers `slider_update`
            curdoc().add_timeout_callback(self.animate, 100)  # 100ms interval
        elif self.step >= self.max_steps:
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
            if self.step in shared_data["epoch"]:
                step_index = shared_data["epoch"].index(self.step)
                xx = shared_data["xx"][step_index]
                yy = shared_data["yy"][step_index]
                zz = shared_data["Z"][step_index]
            if self.step % self.steps == 0:
                new_data = self.source.data.copy()
                shared_data = self.shared_resource.data
                if self.step in shared_data["epoch"]:
                    step_index = shared_data["epoch"].index(self.step)
                    bls = shared_data["bls"][step_index]
                    bpe = shared_data["bpe"][step_index]
                    sensitivity = shared_data["sensitivities"][step_index]
                    softmax_deviations = shared_data["softmax_deviations"][step_index]
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
            shared_data = self.shared_resource.data
            if self.step > 0:
                step_index = shared_data["epoch"].index(self.step-1)
                xx = shared_data["xx"][step_index]
                yy = shared_data["yy"][step_index]
                zz = shared_data["Z"][step_index]
                prev_xs, prev_ys = self.extract_boundary_lines(xx, yy, zz)

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
        self.step = new
        xx, yy, zz = self.calculate_boundaries()
        self.update_boundary(xx, yy, zz)
        self.step_slider.title = f"Epoch: {self.step // 4} Step"

    def get_layout(self):
        return column(
            self.plot,
            self.message_div,
            row(self.play_pause_button, self.backtrack_button, self.proceed_button,),
            row(self.backward_epoch_button, self.forward_epoch_button,),
            row(Spacer(width=50),self.step_slider, Spacer(width=50)),
            row(Div(text="Tracker Colors:"), *self.tracker_buttons),
            row(self.clear_button, self.reset_button)
        )