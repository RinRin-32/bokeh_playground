from bokeh.models import Button, CustomJS, Slider, ColumnDataSource, Div, TapTool, PolySelectTool
from bokeh.layouts import column, row
import numpy as np
import matplotlib
from bokeh.plotting import figure

class LSBoundaryVisualizer:
    def __init__(self, shared_source, shared_resource, max_step, colors, mode='Epoch'):
        self.source = shared_source
        self.shared_resource = shared_resource
        self.max_step = max_step
        self.colors = colors
        self.original_colors = self.source.data['color'].copy() # Store original colors

        self.X = np.column_stack([self.source.data[feature] for feature in self.source.data if feature in ['x', 'y']])
        self.y = self.source.data['class']
        self.classes = np.unique(self.y)

        self.play_pause_button = Button(label="Play")
        self.clear_selection_button = Button(label="Clear Selection")

        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1

        self.plot = figure(
            title="Induced Noise from Adaptive Variation Learning",
            width=400, height=400,
            #sizing_mode="scale_both",
            x_range=(x_min, x_max),
            y_range=(y_min, y_max),
            #tools="tap, box_select, reset, pan, wheel_zoom",
            tools=""
        )

        initial_xs = shared_resource.data["xs"][0]
        initial_ys = shared_resource.data["ys"][0]
        self.boundary_source = ColumnDataSource(data={"xs": initial_xs, "ys": initial_ys})

        self.plot.scatter("x", "y", source=self.source, size="size", color="color", marker="marker", line_color='black', alpha="alpha")
        self.plot.multi_line(xs="xs", ys="ys", source=self.boundary_source, line_width=2, color="black")

        self.step_slider = Slider(start=0, end=self.max_step, value=0, step=1, title=mode)
        self.setup_callbacks()

    def setup_callbacks(self):
        self.step_slider.js_on_change("value", CustomJS(args={"source": self.source, 
                                                               "shared_resource": self.shared_resource,
                                                               "boundary_source": self.boundary_source}, 
        code="""
            var step = cb_obj.value;
            var shared_data = shared_resource.data;
            var step_index = shared_data["epoch"].indexOf(step);
            
            if (step_index !== -1) {
                source.data["size"] = shared_data["size"][step_index];
                source.data["alpha"] = shared_data["alpha"][step_index];
                boundary_source.data["xs"] = shared_data["xs"][step_index];
                boundary_source.data["ys"] = shared_data["ys"][step_index];
                source.data["color"] = shared_data["batch_color"][step_index];

                source.change.emit();
                boundary_source.change.emit();
            }
        """))

        self.play_pause_button.js_on_click(CustomJS(args={"slider": self.step_slider, "button": self.play_pause_button, "max_step": self.max_step}, code="""
            var step = slider.value;
            var is_playing = button.label == "Pause";
            var is_at_end = step >= max_step;
            
            if (is_at_end) {
                button.label = "Restart";
                slider.value = 0;
                step = 0;
            }
            
            if (is_playing) {
                button.label = "Play";
                clearTimeout(slider._timeout);
            } else {
                button.label = "Pause";
                function animate() {
                    if (step < max_step) {
                        step += 1;
                        slider.value = step;
                        slider._timeout = setTimeout(animate, 100);
                    } else {
                        button.label = "Restart";
                    }
                }
                animate();
            }
        """))

        self.source.selected.js_on_change('indices', CustomJS(args={'source': self.source, 'original_colors': self.original_colors}, code="""
            const selected_indices = source.selected.indices;
            const new_colors = source.data['color'].slice(); // Create a copy of the current colors
            for (let i = 0; i < selected_indices.length; i++) {
                new_colors[selected_indices[i]] = 'red';
            }
            source.data['color'] = new_colors;
            source.change.emit();
        """))

        self.clear_selection_button.js_on_click(CustomJS(args={'source': self.source, 'original_colors': self.original_colors}, code="""
            source.data['color'] = original_colors;
            source.selected.indices = [];
            source.change.emit();
        """))

    def get_layout(self):
        return column(self.plot, self.step_slider, self.play_pause_button, self.clear_selection_button)