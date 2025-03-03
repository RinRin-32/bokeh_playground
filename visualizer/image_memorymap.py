from bokeh.models import Button, CustomJS, Slider, ColumnDataSource, Div, HoverTool
from bokeh.layouts import column, row
import numpy as np
import matplotlib
from bokeh.plotting import figure

class ImageSensitivityVisualizer:
    def __init__(self, shared_source, shared_resource, max_epoch, default_color='blue'):
        self.source = shared_source
        self.shared_resource = shared_resource
        self.max_epoch = max_epoch
        self.default_color = default_color

        self.plot = self.create_plot()

        self.step_slider = Slider(start=0, end=self.max_epoch, value=0, step=1, title="Epoch")
        self.play_pause_button = Button(label="Play")
        self.reset_button = Button(label="Reset", button_type="danger")
        self.clear_button = Button(label="Clear", button_type="warning")  # Add Clear button

        self.image_display = Div(
            text="<h3>Selected Images:</h3>", 
            width=300, height=600, 
            stylesheets=[""" .scroll-box { overflow-y: auto; max-height: 600px; padding: 10px; } """], 
            css_classes=["scroll-box"]
        )

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

        self.is_playing = False  # Variable to track whether the animation is playing
        self.step_value = 0  # Track the current step

        self.setup_callbacks()

    def create_plot(self):
        p = figure(title="Memory Map",
                   width=600, height=600,
                   tools='tap,box_select, box_zoom, reset',
                   active_drag='box_select',
                   )
        p.xaxis.axis_label = 'Bayesian Leverage Score'
        p.yaxis.axis_label = 'Bayesian Prediction Error'

        p.scatter(x='bls', y='bpe', color='color', marker='marker', alpha='alpha', size='size', source=self.source)

        p.x_range.only_visible = p.y_range.only_visible = True

        hover = HoverTool(tooltips="""
            <div>
                <img src="data:image/png;base64,@img" width="28" height="28"></img>
                <br>
                <b>Label:</b> @label
            </div>
        """)

        p.add_tools(hover)

        return p
    
    def setup_callbacks(self):
        ## setup all js callbacks here
        self.step_slider.js_on_change("value", CustomJS(args={"source": self.source,
                                                              "shared_resource": self.shared_resource},
        code="""
            var step = cb_obj.value;
            var shared_data = shared_resource.data;
            var step_index = shared_data["epoch"].indexOf(step);
            
            if (step_index !== -1) {
                source.data["bls"] = shared_data["bls"][step_index];
                source.data["bpe"] = shared_data["bpe"][step_index];
                source.change.emit();
            }
        """))

        self.play_pause_button.js_on_click(CustomJS(args={"slider": self.step_slider, "button": self.play_pause_button},
        code="""
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

        self.source.selected.js_on_change("indices", CustomJS(args={"source": self.source, "image_display": self.image_display}, code="""
            var indices = source.selected.indices;
            var images = source.data["img"];
            var labels = source.data["label"];
            
            var html = "<h3>Selected Images:</h3>";
            for (var i = 0; i < indices.length; i++) {
                html += "<div style='display:inline-block; margin:5px; text-align:center;'>";
                html += "<img src='data:image/png;base64," + images[indices[i]] + "' width='64' height='64'><br>";
                html += "Label: " + labels[indices[i]] + "</div>";
            }
            image_display.text = '<div class="scroll-box">' + html + '</div>';
        """))

    def get_layout(self):
        return column(
            row(self.plot, self.image_display,),
            row(self.play_pause_button, self.reset_button, self.clear_button),
            row(self.step_slider),
            row(Div(text="Tracker Colors:"), *self.tracker_buttons)
        )