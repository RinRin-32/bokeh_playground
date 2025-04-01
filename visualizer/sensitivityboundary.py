from bokeh.models import Button, CustomJS, Slider, ColumnDataSource, Div
from bokeh.layouts import column, row
import numpy as np
import matplotlib
from bokeh.plotting import figure

class EvolvingBoundaryVisualizer:
    def __init__(self, shared_source, shared_resource, steps, colors, batches=4, max_steps=30, show_lambda=False):
        self.source = shared_source
        self.shared_resource = shared_resource
        self.batches = batches
        self.steps = steps
        self.colors = colors
        self.max_steps = max_steps
        self.show_lambda = show_lambda

        self.X = np.column_stack([self.source.data[feature] for feature in self.source.data if feature in ['x', 'y']])
        self.y = self.source.data['class']
        self.classes = np.unique(self.y)

        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1

        self.message_div = Div(text="", width=400, height=50)
        self.epoch_div = Div(text=f"Epoch: 0", width=60, height=20)  # New Div to display Epoch

        self.plot = figure(
            title="Sensitivity over Training",
            width=600, height=600,
            x_range=(x_min, x_max),
            y_range=(y_min, y_max),
            tools="tap,box_select,box_zoom,reset,pan",
            active_drag="box_select"
        )

        initial_xs = shared_resource.data["xs"][0]
        initial_ys = shared_resource.data["ys"][0]
        prev_xs = shared_resource.data["xs"][0] if len(shared_resource.data["xs"]) < 2 else shared_resource.data["xs"][1]
        prev_ys = shared_resource.data["ys"][0] if len(shared_resource.data["ys"]) < 2 else shared_resource.data["ys"][1]

        self.boundary_source = ColumnDataSource(data={"xs": initial_xs, "ys": initial_ys, "prev_xs": prev_xs, "prev_ys": prev_ys})

        self.plot.multi_line(xs="prev_xs", ys="prev_ys", source=self.boundary_source, line_width=2, color="grey")
        self.plot.multi_line(xs="xs", ys="ys", source=self.boundary_source, line_width=2, color="black")

        self.plot.scatter("x", "y", source=self.source, size="size", color="color", marker="marker", alpha="alpha",  line_color='black')

        self.step_slider = Slider(start=0, end=self.max_steps, value=0, step=1, title="Step")
        self.play_pause_button = Button(label="Play")

        self.is_playing = False  # Variable to track whether the animation is playing
        self.step_value = 0  # Track the current step

        self.setup_callbacks()

    def setup_callbacks(self):
        # Existing step slider callback
        self.step_slider.js_on_change("value", CustomJS(args={"source": self.source, 
                                                            "shared_resource": self.shared_resource,
                                                            "boundary_source": self.boundary_source,
                                                            "epoch_div": self.epoch_div,  # Pass the epoch Div
                                                            "batches": self.batches, "condition": self.show_lambda}, 
        code="""
            var step = cb_obj.value;
            var shared_data = shared_resource.data;
            var step_index = shared_data["step"].indexOf(step);
            
            if (step_index !== -1) {
                source.data["alpha"] = shared_data["alpha"][step_index];
                source.data["size"] = shared_data["size"][step_index];

                boundary_source.data["xs"] = shared_data["xs"][step_index];
                boundary_source.data["ys"] = shared_data["ys"][step_index];
                boundary_source.data["prev_xs"] = shared_data["xs"][step_index];
                boundary_source.data["prev_ys"] = shared_data["ys"][step_index];

                if (step_index > 0) {
                    boundary_source.data["prev_xs"] = shared_data["xs"][step_index - 1];
                    boundary_source.data["prev_ys"] = shared_data["ys"][step_index - 1];
                }

                source.change.emit();
                boundary_source.change.emit();

                // Update the epoch display
                var epoch = Math.floor(step / batches);
                epoch_div.text = "Epoch: " + epoch;
            }
        """))

        # Update play/pause button behavior
        self.play_pause_button.js_on_click(CustomJS(args={"slider": self.step_slider, "button": self.play_pause_button}, code="""
            var step = slider.value;
            var is_playing = button.label == "Pause";  // Check if currently playing
            
            if (is_playing) {
                button.label = "Play";  // Change to "Play" when pausing
                clearTimeout(slider._timeout);
            } else {
                button.label = "Pause";  // Change to "Pause" when playing
                function animate() {
                    if (step < slider.end) {
                        step += 1;
                        slider.value = step;
                        slider._timeout = setTimeout(animate, 100);
                    }
                }
                animate();
            }
        """))

    def get_layout(self):
        return column(
            self.plot,
            self.message_div,
            self.play_pause_button,
            row(self.epoch_div, self.step_slider),
        )