from bokeh.models import ColumnDataSource, CustomJS, Slider, Button, HoverTool, ImageURL
from bokeh.layouts import column, row
from bokeh.plotting import figure

class Sample:
    def __init__(self, shared_source, shared_resource, plot_name, y_range, n_sample, max_epoch, default_color='white', display_mode='row'):
        self.shared_source = shared_source
        self.shared_resource = shared_resource
        self.plot_name = plot_name
        self.n_sample = n_sample
        self.max_epoch = max_epoch
        self.default_color = default_color
        self.display_mode = display_mode  # "column" or "row"

        self.play_pause_button = Button(label="Play")

        self.y_min = y_range[0]
        self.y_max = y_range[1]

        # Slider to control the epoch
        self.step_slider = Slider(start=0, end=self.max_epoch, value=0, step=1, title="Epoch")

        # Set up Bokeh plot
        self.plot = figure(title="Image Display", tools="pan, wheel_zoom, box_zoom, reset", x_range=(0, 10), y_range=(0, 10))

        # Set up the plot data using pre-calculated positions
        self.source = ColumnDataSource(self.shared_source.data)

        # Set up callbacks for slider updates
        self.setup_callbacks()

        # Add image display glyph
        self.plot.image_url(url="img", x="x", y="y", source=self.source, height=0.1, width=0.1)

        # Add hover tool to display additional information on hover
        hover = HoverTool()
        hover.tooltips = [("Label", "@label"), ("Noise", "@y")]
        self.plot.add_tools(hover)

    def setup_callbacks(self):
        # Update plot data when the slider value changes
        self.step_slider.js_on_change("value", CustomJS(args={
            "original": self.shared_resource, 
            "source": self.shared_source,
            "plot": self.plot,
        },
        code="""
            var step = cb_obj.value;
            var shared_data = original.data;
            source.data["y"] = shared_data["y"][step];
            source.data["x"] = shared_data["x"][step];
            source.data["img"] = shared_data["img"][step];
            source.data["label"] = shared_data["label"][step];
            source.change.emit();
        """))

        # Play/pause button functionality
        self.play_pause_button.js_on_click(CustomJS(args={"slider": self.step_slider, "button": self.play_pause_button, "max_epoch": self.max_epoch}, code="""
            var step = slider.value;
            var is_playing = button.label == "Pause";
            var is_at_end = step >= max_epoch;
            
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
                    if (step < max_epoch) {
                        step += 1;
                        slider.value = step;
                        slider._timeout = setTimeout(animate, 500);
                    } else {
                        button.label = "Restart";
                    }
                }
                animate();
            }
        """))

    def get_layout(self):
        # Layout: Bokeh plot for images and slider below
        if self.display_mode == "row":
            return column(row(self.plot), self.step_slider, self.play_pause_button)
        else:
            return column(self.plot, self.step_slider, self.play_pause_button)