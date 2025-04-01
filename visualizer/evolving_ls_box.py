from bokeh.models import ColumnDataSource, CustomJS, Div, HoverTool, Quad, Slider, Button, TapTool, BoxSelectTool, BoxEditTool
from bokeh.plotting import figure
from bokeh.layouts import row, column

class EvolvingLabelNoisePlot:
    def __init__(self, shared_source, shared_resource, plot_name, y_range, n_sample, max_epoch, default_color='white'):
        self.shared_source = shared_source
        self.shared_resource = shared_resource
        self.plot_name = plot_name
        self.n_sample = n_sample
        self.max_epoch = max_epoch
        self.playing = False
        self.default_color = default_color

        self.y_min = y_range[0]
        self.y_max = y_range[1]

        self.selection_box_source = ColumnDataSource(data=dict(left=[], right=[], bottom=[], top=[]))

        self.step_slider = Slider(start=0, end=self.max_epoch, value=0, step=1, title="Epoch")
        self.play_button = Button(label="Play", button_type="success")

        self.plot = self.create_plot()
        self.image_display = self.create_image_display()

        self.setup_callbacks()

        taptool = self.plot.select(dict(type=TapTool))
        taptool.callback = CustomJS(args=dict(source=self.shared_source), code="""
            var indices = source.selected.indices;
            if (indices.length > 1) {
                source.selected.indices = [indices[indices.length - 1]];  // Keep only the last selected point
                source.change.emit();  // Update the selection
            }
        """)


    def create_plot(self):
        p = figure(
            width=800, height=500,
            tools="reset,save",
            title=f"{self.plot_name} Label Noise Distribution",
            x_axis_label="Sorted Sample By Noise", y_axis_label=r"Label Noise ||ε||₂",
            x_range=(-10, self.n_sample),
            y_range=(self.y_min-0.05, self.y_max)
        )

        p.title.text_font_size = "25px"
        p.title.align = 'center'
        p.yaxis.major_label_text_font_size = "15pt"
        p.yaxis.axis_label_text_font_size = "20pt"

        # Create a source for the rectangles (boxes)
        box_source = ColumnDataSource(data=dict(
            left=[0, 800], right=[100, 1000],  # X positions for two boxes
            bottom=[0.05, -0.01], top=[0.5, 0.2], color=['yellow', 'blue']  # Y positions for two boxes
        ))

        quad_renderer = p.quad(left="left", right="right", bottom="bottom", top="top", 
                       source=box_source, fill_alpha=0.3, color="color")

        # Add BoxEditTool
        box_edit = BoxEditTool(renderers=[quad_renderer])
        
        p.add_tools(box_edit)

        # Scatter plot
        p.scatter("x", "y", source=self.shared_source, size="size", color="color", 
                legend_label="Data", fill_alpha=0.6, line_color='black')

        return p

    def create_image_display(self):
        return Div(
            text="<h3>Selected Images:</h3>",
            sizing_mode="stretch_width", height=500,
            stylesheets=[""" .scroll-box { overflow-y: auto; max-height: 500px; padding: 10px; } """],
            css_classes=["scroll-box"]
        )

    def setup_callbacks(self):
        self.step_slider.js_on_change("value", CustomJS(args={"original": self.shared_resource,
                                                              "intermediate": self.shared_source},
        code="""
            var step = cb_obj.value;
            var shared_data = original.data;
            
            intermediate.data["y"] = shared_data["y"][step];
            intermediate.data["x"] = shared_data["x"][step];
            //intermediate.data["noise_chart"] = shared_data["noise_chart"][step];

            intermediate.change.emit();
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

        # Trigger selection when data updates
        self.shared_source.js_on_change("data", CustomJS(args={
            "source": self.shared_source, "image_display": self.image_display,
            "selection_box": self.selection_box_source
        }, code="""
            var indices = source.selected.indices;
            var images = source.data["img"];
            var labels = source.data["label"];
            //var noise_charts = source.data["noise_chart"];
            var html = "";

            if (indices.length === 0) {
                html += "<h3>No selection made.</h3>";
            } else {
                // Selection exists: Show selected images
                var groupedImages = {};
                html += "<h3>Selected Images:</h3>";
                
                for (var i = 0; i < indices.length; i++) {
                    var label = labels[indices[i]];
                    var imgTag = "<img src='data:image/png;base64," + images[indices[i]] + "' width='64' height='64'>";
                    //var noiseChartTag = "<img src='data:image/png;base64," + noise_charts[indices[i]] + "' width='150' height='100'>";
                    //var wrappedTag = "<span style='display: inline-block; margin: 5px;'>" + imgTag + noiseChartTag + "</span>";
                    var wrappedTag = "<span style='display: inline-block; margin: 5px;'>" + imgTag + "</span>";

                    if (!(label in groupedImages)) {
                        groupedImages[label] = [];
                    }
                    groupedImages[label].push(wrappedTag);
                }

                var sortedLabels = Object.keys(groupedImages).sort();
                sortedLabels.forEach(function(label) {
                    html += "<div style='margin-bottom:10px; max-width: 100%;'><b>Label: " + label + "</b><br>";
                    html += "<div style='display: flex; flex-wrap: wrap; gap: 5px;'>" + groupedImages[label].join("") + "</div></div>";
                });
            }

            image_display.text = '<div class="scroll-box">' + html + '</div>';
        """))

        # Also trigger when selection changes
        self.shared_source.selected.js_on_change("indices", CustomJS(args={
            "source": self.shared_source, "image_display": self.image_display,
            "selection_box": self.selection_box_source
        }, code="""
            var indices = source.selected.indices;
            var images = source.data["img"];
            var labels = source.data["label"];
            //var noise_charts = source.data["noise_chart"];
            var html = "";

            if (indices.length === 0) {
                html += "<h3>No selection made.</h3>";
            } else {
                // Selection exists: Show selected images
                var groupedImages = {};
                html += "<h3>Selected Images:</h3>";
                
                for (var i = 0; i < indices.length; i++) {
                    var label = labels[indices[i]];
                    var imgTag = "<img src='data:image/png;base64," + images[indices[i]] + "' width='64' height='64'>";
                    //var noiseChartTag = "<img src='data:image/png;base64," + noise_charts[indices[i]] + "' width='150' height='100'>";
                    //var wrappedTag = "<span style='display: inline-block; margin: 5px;'>" + imgTag + noiseChartTag + "</span>";
                    var wrappedTag = "<span style='display: inline-block; margin: 5px;'>" + imgTag + "</span>";

                    if (!(label in groupedImages)) {
                        groupedImages[label] = [];
                    }
                    groupedImages[label].push(wrappedTag);
                }

                var sortedLabels = Object.keys(groupedImages).sort();
                sortedLabels.forEach(function(label) {
                    html += "<div style='margin-bottom:10px; max-width: 100%;'><b>Label: " + label + "</b><br>";
                    html += "<div style='display: flex; flex-wrap: wrap; gap: 5px;'>" + groupedImages[label].join("") + "</div></div>";
                });
            }

            image_display.text = '<div class="scroll-box">' + html + '</div>';
        """))

    def get_layout(self):
        return row(column(self.plot, self.step_slider, self.play_button), self.image_display)