from bokeh.models import Button, CustomJS, Slider, ColumnDataSource, Div, Spacer
from bokeh.layouts import column, row
import numpy as np
import matplotlib
from bokeh.plotting import figure

class TestNLLAnimation:
    def __init__(self, shared_source, shared_resource, max_epoch, subsample_intermediate, subsample_source, default_color='blue'):
        self.source = shared_source
        self.shared_resource = shared_resource
        self.max_epoch = max_epoch
        self.playing = False
        self.default_color = default_color
        self.subsample_intermediate = subsample_intermediate
        self.subsample_source = subsample_source

        initial_epoch = 0
        initial_index = shared_resource.data["epoch"].index(initial_epoch) if initial_epoch in shared_resource.data["epoch"] else None

        if initial_index is not None:
            self.data_stream = ColumnDataSource(data={
                'epoch': [shared_resource.data["epoch"][initial_index]],
                'test_nll': [shared_resource.data["test_nll"][initial_index]],
                'estimated_nll': [shared_resource.data["estimated_nll"][initial_index]]
            })
        else:
            self.data_stream = ColumnDataSource(data={'epoch': [], 'test_nll': [], 'estimated_nll': []})

        self.step_slider = Slider(start=0, end=self.max_epoch, value=0, step=1, title="Epoch")
        self.play_button = Button(label="Play", button_type="success")

        self.plot = self.create_plot()

        self.tracker_colors = ["#d55e00", "#cc79a7", "#0072b2", "#f0e442", "#009e73"]
        self.tracker_colors_hex = [matplotlib.colors.rgb2hex(c) for c in self.tracker_colors]  # Store as hex

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

        self.clear_button = Button(label="Clear", button_type="warning")

        self.setup_callbacks()

    def setup_callbacks(self):
        self.step_slider.js_on_change("value", CustomJS(args={"source": self.data_stream,
                                                              "original": self.shared_resource,
                                                              "intermediate": self.source,
                                                              "subsample_intermediate": self.subsample_intermediate,
                                                              "subsample_source": self.subsample_source},
        code="""
            var step = cb_obj.value;
            var shared_data = original.data;
            var current_epochs = source.data["epoch"];
            
            if (current_epochs.length > 0) {
                var prev_max = Math.max(...current_epochs);
                
                if (step > prev_max) {
                    for (var s = prev_max + 1; s <= step; s++) {
                        var step_index = shared_data["epoch"].indexOf(s);
                        if (step_index !== -1) {
                            source.data["epoch"].push(shared_data["epoch"][step_index]);
                            source.data["test_nll"].push(shared_data["test_nll"][step_index]);
                            source.data["estimated_nll"].push(shared_data["estimated_nll"][step_index]);
                        }
                    }
                } else if (step < prev_max) {
                    while (source.data["epoch"].length > 0 && Math.max(...source.data["epoch"]) > step) {
                        source.data["epoch"].pop();
                        source.data["test_nll"].pop();
                        source.data["estimated_nll"].pop();
                    }
                }
            } else {
                var step_index = shared_data["epoch"].indexOf(step);
                if (step_index !== -1) {
                    source.data["epoch"].push(shared_data["epoch"][step_index]);
                    source.data["test_nll"].push(shared_data["test_nll"][step_index]);
                    source.data["estimated_nll"].push(shared_data["estimated_nll"][step_index]);
                }
            }
            intermediate.data["y"] = shared_data["y"][step];
            intermediate.data["x"] = shared_data["x"][step];
            intermediate.data["noise_chart"] = shared_data["noise_chart"][step];

            for (let i = 0; i < subsample_intermediate.length; i++) {
                subsample_intermediate[i].data = subsample_source[step][i].data;  // Update bar chart data
                subsample_intermediate[i].change.emit();
                console.log("Step: ", step);
            }

            intermediate.change.emit();
            source.change.emit();
        """))

        self.play_button.js_on_event("button_click", CustomJS(args={"slider": self.step_slider,
                                                                     "button": self.play_button,
                                                                     "max_epoch": self.max_epoch},
        code="""
            if (button.label === "Play") {
                button.label = "Pause";
                var step = slider.value;
                var interval = setInterval(function() {
                    if (slider.value < max_epoch && button.label === "Pause") {
                        slider.value += 1;
                    } else {
                        clearInterval(interval);
                        button.label = "Play";
                    }
                }, 500);
                button.interval = interval;
            } else {
                clearInterval(button.interval);
                button.label = "Play";
            }
        """))

        self.clear_button.js_on_click(CustomJS(args={"source": self.source, "default_color": self.default_color}, code="""
            var new_data = source.data;
            for (var idx = 0; idx < new_data["color"].length; idx++) {
                new_data["color"][idx] = default_color
                new_data["alpha"][idx] = 1.0;
                new_data["size"][idx] = 6;  // Reset size
            }
            source.selected.indices = [];  // Clear selection
            source.change.emit();
        """)) 

    def create_plot(self):
        p = figure(title="Test vs Estimate NLL", width=600, height=300, x_range=(0, self.max_epoch), tools="")
        p.xaxis.axis_label = 'Epoch'
        p.yaxis.axis_label = '\n\nNLL'
        p.line(x='epoch', y='test_nll', color='black', legend_label="Test NLL", source=self.data_stream)
        p.line(x='epoch', y='estimated_nll', color='red', legend_label="Estimated NLL", source=self.data_stream)
        return p

    def get_layout(self):
        return column(
            self.plot, 
            row(Spacer(width=30), self.step_slider, self.play_button, self.clear_button),
            row(Spacer(width=30), Div(text="Tracker Colors:"), *self.tracker_buttons)
        )