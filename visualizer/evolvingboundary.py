from bokeh.models import Button, CustomJS, Slider, ColumnDataSource, Div
from bokeh.layouts import column, row
import numpy as np
import matplotlib
from bokeh.plotting import figure

class EvolvingBoundaryVisualizer:
    def __init__(self, shared_source, shared_resource, steps, colors, batches=4, max_steps=30):
        self.source = shared_source
        self.shared_resource = shared_resource
        self.batches = batches
        self.steps = steps
        self.colors = colors
        self.max_steps = max_steps

        self.X = np.column_stack([self.source.data[feature] for feature in self.source.data if feature in ['x', 'y']])
        self.y = self.source.data['class']
        self.classes = np.unique(self.y)

        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1

        self.message_div = Div(text="", width=400, height=50)
        self.epoch_div = Div(text=f"Epoch: 0", width=60, height=20)  # New Div to display Epoch

        self.plot = figure(
            title="Evolving Boundary Visualization",
            width=600, height=600,
            x_range=(x_min, x_max),
            y_range=(y_min, y_max),
            tools="tap,box_select,box_zoom,reset,pan",
            active_drag="box_select"
        )

        # Initialize boundary source with data from step 0
        initial_xs = shared_resource.data["xs"][0]
        initial_ys = shared_resource.data["ys"][0]
        self.boundary_source = ColumnDataSource(data={"xs": initial_xs, "ys": initial_ys, "prev_xs": initial_xs, "prev_ys": initial_ys})

        self.plot.scatter("x", "y", source=self.source, size="size", color="color", marker="marker", alpha="alpha")
        self.plot.multi_line(xs="prev_xs", ys="prev_ys", source=self.boundary_source, line_width=2, color="grey")
        self.plot.multi_line(xs="xs", ys="ys", source=self.boundary_source, line_width=2, color="black")

        self.step_slider = Slider(start=0, end=self.max_steps, value=0, step=1, title="Step")
        self.play_pause_button = Button(label="Play")
        self.reset_button = Button(label="Reset", button_type="danger")
        self.clear_button = Button(label="Clear", button_type="warning")  # Add Clear button

        self.tracker_colors = ["#d55e00", "#cc79a7", "#0072b2", "#f0e442", "#009e73"]
        self.tracker_colors_hex = [matplotlib.colors.rgb2hex(c) for c in self.tracker_colors]  # Store as hex

        # Forward/Backward buttons for step
        self.forward_step_button = Button(label="Forward Step", width=150, button_type="success")
        self.backward_step_button = Button(label="Backward Step", width=150, button_type="warning")

        # Forward/Backward buttons for epoch
        self.forward_epoch_button = Button(label="Forward Epoch", width=150, button_type="success")
        self.backward_epoch_button = Button(label="Backward Epoch", width=150, button_type="warning")

        # Tracker buttons setup
        self.tracker_buttons = []
        for i, color in enumerate(self.tracker_colors_hex):
            style_btn = f"""
            .bk-btn {{
                color: {color};
                background-color: {color};
            }}
            .bk-btn:hover {{
                background-color: {color};
                opacity: 0.8; /* Optional: Adds a slight transparency effect on hover */
            }}
            """
            
            button = Button(label=f"", width=50, height=50, stylesheets=[style_btn], css_classes=[f'color-button-{i}'])
            button_callback = CustomJS(args={"source": self.source, "color": color, "all_color": self.tracker_colors}, code="""
                var selected_indices = source.selected.indices;
                if (selected_indices.length == 0) {
                    alert("No points selected to apply color.");
                    return;
                }

                var new_data = source.data;
                for (var idx = 0; idx < new_data["color"].length; idx++) {
                    if (selected_indices.includes(idx)) {
                        new_data["color"][idx] = color;
                        new_data["alpha"][idx] = 1.0;
                        new_data["size"][idx] = 10;
                    } else if (new_data["color"][idx] != "grey" && new_data["color"][idx] != color && !all_color.includes(new_data["color"][idx])) {
                        new_data["color"][idx] = "grey";
                        new_data["alpha"][idx] = 0.2;
                    }
                }
                source.change.emit();
            """)
            button.js_on_click(button_callback)
            self.tracker_buttons.append(button)

        self.is_playing = False  # Variable to track whether the animation is playing
        self.step_value = 0  # Track the current step

        self.setup_callbacks()

    def setup_callbacks(self):
        # Existing step slider callback
        self.step_slider.js_on_change("value", CustomJS(args={"source": self.source, 
                                                            "shared_resource": self.shared_resource,
                                                            "boundary_source": self.boundary_source,
                                                            "epoch_div": self.epoch_div,  # Pass the epoch Div
                                                            "batches": self.batches}, 
        code="""
            var step = cb_obj.value;
            var shared_data = shared_resource.data;
            var step_index = shared_data["step"].indexOf(step);
            
            if (step_index !== -1) {
                source.data["bls"] = shared_data["bls"][step_index];
                source.data["bpe"] = shared_data["bpe"][step_index];
                source.data["sensitivities"] = shared_data["sensitivities"][step_index];
                source.data["softmax_deviations"] = shared_data["softmax_deviations"][step_index];

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

        self.reset_button.js_on_click(CustomJS(args={"slider": self.step_slider}, code="""
            slider.value = 0;
        """))

        # Clear button JS callback
        self.clear_button.js_on_click(CustomJS(args={"source": self.source, "colors": self.colors}, code="""
            var new_data = source.data;
            for (var idx = 0; idx < new_data["color"].length; idx++) {
                var class_idx = new_data["class"][idx];
                new_data["color"][idx] = colors[class_idx];  // Reset to original color based on class
                new_data["alpha"][idx] = 1.0;
                new_data["size"][idx] = 6;  // Reset size
            }
            source.selected.indices = [];  // Clear selection
            source.change.emit();
        """))

        self.forward_step_button.js_on_click(CustomJS(args={"slider": self.step_slider}, code="""
            var step = slider.value;
            if (step < slider.end) {
                slider.value = step + 1;
            }
        """))

        self.backward_step_button.js_on_click(CustomJS(args={"slider": self.step_slider}, code="""
            var step = slider.value;
            if (step > slider.start) {
                slider.value = step - 1;
            }
        """))

        self.forward_epoch_button.js_on_click(CustomJS(args={"slider": self.step_slider}, code="""
            var step = slider.value;
            if (step + 4 <= slider.end) {
                slider.value = step + 4;
            }
        """))

        self.backward_epoch_button.js_on_click(CustomJS(args={"slider": self.step_slider}, code="""
            var step = slider.value;
            if (step - 4 >= slider.start) {
                slider.value = step - 4;
            }
        """))

    def get_layout(self):
        return column(
            self.plot,
            self.message_div,
            row(self.play_pause_button, self.reset_button, self.clear_button),
            row(self.epoch_div, self.step_slider),
            row(self.backward_step_button, self.forward_step_button),
            row(self.backward_epoch_button, self.forward_epoch_button),
            row(Div(text="Tracker Colors:"), *self.tracker_buttons)
        )