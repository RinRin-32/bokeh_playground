from bokeh.models import Button, CustomJS, Slider, ColumnDataSource, Div, TapTool, PointDrawTool
from bokeh.layouts import column, row
import numpy as np
from bokeh.plotting import figure
import matplotlib

class LSBoundaryVisualizer:
    def __init__(self, shared_source, shared_resource, max_step, colors, total_batches, mode='Step', sig_projection=False, barplot_shared_source=None, barplot_shared_resource=None, barplot_module=None):
        self.source = shared_source
        self.shared_resource = shared_resource
        self.max_step = max_step
        self.max_epoch = total_batches
        self.colors = colors
        self.original_colors = shared_source.data['color'].copy() # Store original colors
        self.total_batches = total_batches
        self.toggle = sig_projection
        self.barplot_source = barplot_shared_source
        self.barplot_resource = barplot_shared_resource
        self.barmode = (self.barplot_source != None)
        self.barplot = barplot_module

        self.tracker_colors = ["#d55e00", "#cc79a7", "#0072b2", "#f0e442", "#009e73"]
        self.tracker_colors_hex = [matplotlib.colors.rgb2hex(c) for c in self.tracker_colors]  # Store as hex

        self.X = np.column_stack([self.source.data[feature] for feature in self.source.data if feature in ['x', 'y']])
        self.y = self.source.data['class']
        self.classes = np.unique(self.y)

        self.play_pause_button = Button(label="Play")
        self.clear_selection_button = Button(label="Clear Selection")
        self.epoch_display = Div(text=f"Epoch: 0", width=60, height=20) # Added epoch display

        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1

        self.plot = figure(
            title="Adaptive Label Noise from Variational Learning",
            width=600, height=600,
            #sizing_mode="scale_both",
            x_range=(x_min, x_max),
            y_range=(y_min, y_max),
            tools=" ",
            #active_drag="box_select",
        )

        initial_xs = shared_resource.data["xs"][0]
        initial_ys = shared_resource.data["ys"][0]
        self.boundary_source = ColumnDataSource(data={"xs": initial_xs, "ys": initial_ys})

        #self.plot.scatter("x", "y", source=self.source, size="size", color="color", marker="marker", line_color='black', alpha="alpha")
        self.plot.scatter("x", "y", source=self.source, size=8, color="color", marker="marker", line_color='black', alpha=1)
        self.plot.multi_line(xs="xs", ys="ys", source=self.boundary_source, line_width=6, color="black")

        self.step_slider = Slider(start=0, end=self.max_step, value=0, step=1, title=mode)

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
                        new_data["selection"][idx] = 15
                        new_data["bar_alpha"][idx] = 1.0
                        new_data["temp"][idx] = new_data["alpha"][idx]
                        new_data["alpha"][idx] = 1.0
                    }
                }
                source.change.emit();
            """)
            button.js_on_click(button_callback)
            self.tracker_buttons.append(button)

        taptool = self.plot.select(dict(type=TapTool))
        taptool.callback = CustomJS(args=dict(source=self.source), code="""
            var indices = source.selected.indices;
            if (indices.length > 1) {
                source.selected.indices = [indices[indices.length - 1]];  // Keep only the last selected point
                source.change.emit();  // Update the selection
            }
        """)

        # Identify the point with the maximum noise (most positive)
        max_noise_idx = np.argmax(self.source.data["noise"])
        max_noise_point = self.X[max_noise_idx]
        max_noise_class = self.y[max_noise_idx]

        # Identify the point with the minimum noise (most negative)
        min_noise_idx = np.argmin(self.source.data["noise"])
        min_noise_point = self.X[min_noise_idx]
        min_noise_class = self.y[min_noise_idx]

        # Make sure they are different classes (if not, find alternative min from another class)
        if max_noise_class == min_noise_class:
            # Search for min from a different class than the max_noise_class
            mask = self.y != max_noise_class
            candidate_noises = np.array(self.source.data["noise"])[mask]
            candidate_points = self.X[mask]
            candidate_classes = self.y[mask]
            min_candidate_idx = np.argmin(candidate_noises)
            min_noise_point = candidate_points[min_candidate_idx]
            min_noise_class = candidate_classes[min_candidate_idx]

        # Now use these two points to set the line
        self.fixed_line_start = min_noise_point
        self.fixed_line_end = max_noise_point

        # Compute the direction vector
        direction = self.fixed_line_end - self.fixed_line_start
        direction = direction / np.linalg.norm(direction)

        # Get the bounds of the plot from your data
        x_vals = self.X[:, 0]
        y_vals = self.X[:, 1]
        x_min, x_max = x_vals.min(), x_vals.max()
        y_min, y_max = y_vals.min(), y_vals.max()

        # Determine how far to extend (pick large enough range to cover the plot)
        max_range = max(x_max - x_min, y_max - y_min)
        extend_length = max_range * 0.7  # or tweak this scalar as needed

        # Extend in both directions from the midpoint of the two original points
        midpoint = (self.fixed_line_start + self.fixed_line_end) / 2

        extended_start = midpoint - direction * extend_length
        extended_end = midpoint + direction * extend_length

        # Create source for fixed line
        self.line_source = ColumnDataSource(data={
            "xs": [[extended_start[0], extended_end[0]]],
            "ys": [[extended_start[1], extended_end[1]]]
        })

        # Add fixed line to plot
        self.plot.multi_line(xs="xs", ys="ys", source=self.line_source, line_width=2, color="red", line_dash="dashed")

        # Source for the point on the line
        self.point_source = ColumnDataSource(data={"x": [extended_start[0]], "y": [extended_start[1]]})
        self.plot.circle(x="x", y="y", size=10, color="red", source=self.point_source)

        # Slider to move point along the line (0=start, 1=end)
        self.line_slider = Slider(start=0, end=1, value=0, step=0.01, title=None, show_value=False)

        self.slider_title = Div(text="<b>Move Point on Line</b>")

        self.line_slider.js_on_change("value", CustomJS(args=dict(
            point_source=self.point_source,
            x0=extended_start[0], y0=extended_start[1],
            x1=extended_end[0], y1=extended_end[1],
            source=self.source,
            bar_plot=self.barplot_source,
            dist_source=self.barplot.dist_source,
            marker_source=self.barplot.marker_source
        ), code="""
            const t = cb_obj.value;
            const px = (1 - t) * x0 + t * x1;
            const py = (1 - t) * y0 + t * y1;

            // Update red dot
            point_source.data.x[0] = px;
            point_source.data.y[0] = py;
            point_source.change.emit();

            const data = source.data;
            const bar_data = bar_plot.data;
            const EPS = 0.05;

            for (let i = 0; i < data['color'].length; i++) {
                data['color'][i] = data['original_color'][i];
            }
            for (let i = 0; i < bar_data['color'].length; i++) {
                bar_data['color'][i] = 'white';
            }

            let selected_region = null;
            for (let i = 0; i < data['x'].length; i++) {
                const dx = Math.abs(data['x'][i] - px);
                const dy = Math.abs(data['y'][i] - py);
                if (dx <= EPS && dy <= EPS) {
                    data['color'][i] = 'red';
                    selected_region = data['region'][i];
                }
            }

            if (selected_region !== null) {
                for (let i = 0; i < bar_data['region'].length; i++) {
                    if (bar_data['region'][i] === selected_region) {
                        bar_data['color'][i] = 'red';
                        break;
                    }
                }
            }

            // Update marker on smoothed line (find closest x)
            const xs = dist_source.data.x;
            const ys = dist_source.data.y;
            let closest_i = 0;
            let closest_dist = Math.abs(xs[0] - px);
            for (let i = 1; i < xs.length; i++) {
                const dist = Math.abs(xs[i] - px - 1);
                if (dist < closest_dist) {
                    closest_i = i;
                    closest_dist = dist;
                }
            }

            marker_source.data.x[0] = xs[closest_i];
            marker_source.data.y[0] = ys[closest_i];
            marker_source.change.emit();

            source.change.emit();
            bar_plot.change.emit();
        """))

        self.setup_callbacks()

    def setup_callbacks(self):
        self.step_slider.js_on_change("value", CustomJS(args={"source": self.source, 
                                                               "shared_resource": self.shared_resource,
                                                               "boundary_source": self.boundary_source,
                                                               "epoch_display": self.epoch_display,
                                                               "total_batches": self.total_batches,
                                                               "toggle": self.toggle,
                                                               "barplot": self.barmode,
                                                               "bar_resource": self.barplot_resource,
                                                               "bar_source": self.barplot_source
                                                               }, 
        code="""
            var step = cb_obj.value;
            var shared_data = shared_resource.data;
            var step_index = step
            var current_epoch = Math.floor(step / total_batches);
            epoch_display.text = "Epoch: " + current_epoch;
            
            if (step_index != -1){
                source.data["size"] = shared_data["size"][step_index];
                source.data["alpha"] = shared_data["alpha"][step_index];
                boundary_source.data["xs"] = shared_data["xs"][step_index];
                boundary_source.data["ys"] = shared_data["ys"][step_index];

                if (toggle){
                    source.data["logits"] = shared_data["logits"][step_index];
                    source.data["sig_in"] = shared_data["sig_in"][step_index];
                    source.data["noise"] = shared_data["noise"][step_index];
                }
                
                if (barplot){
                    bar_source.data["noise"] = bar_resource.data["noise"][step_index];
                    bar_source.data["sig_in"] = bar_resource.data["sig_in"][step_index];
                    bar_source.change.emit();
                }

                source.change.emit();
                boundary_source.change.emit();
            }
        """))

        self.play_pause_button.js_on_click(CustomJS(args={"slider": self.step_slider, "button": self.play_pause_button, "max_step": self.max_step, "epoch_display": self.epoch_display, "total_batches": self.total_batches}, code="""
            var step = slider.value;
            var is_playing = button.label == "Pause";
            var is_at_end = step >= max_step;
            var current_epoch = Math.floor(step / total_batches);
            epoch_display.text = "Epoch: " + current_epoch;
            
            if (is_at_end) {
                button.label = "Restart";
                slider.value = 0;
                step = 0;
                epoch_display.text = "Epoch: 0";
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
                        var current_epoch = Math.floor(step / total_batches);
                        epoch_display.text = "Epoch: " + current_epoch;
                        slider._timeout = setTimeout(animate, 100);
                    } else {
                        button.label = "Restart";
                    }
                }
                animate();
            }
        """))

        self.clear_selection_button.js_on_click(CustomJS(args={'source': self.source}, code="""
            var data = source.data;
            var color = "white";  // Set the color to white
            // Set all points to white
            for (var i = 0; i < data['color'].length; i++) {
                data['color'][i] = color;
                data['bar_alpha'][i] = 0;
                data['selection'][i] = 6;
                if (data['temp'][i] != 0){
                    data['alpha'][i] = data['temp'][i]
                    data['temp'][i] = 0
                }
            }
            source.selected.indices = [];  // Clear selection
            source.change.emit();  // Notify the source to update the plot
        """))

    def get_layout(self):
        return column(
            self.plot, 
            #self.epoch_display, 
            #self.step_slider, 
            #self.play_pause_button, 
            #self.clear_selection_button,
            self.slider_title,
            self.line_slider,
            #row(Div(text="Tracker Colors:"), *self.tracker_buttons)
            )