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
        self.image_displays = self.create_image_displays()

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
            x_range=((-1*0.05*self.n_sample), self.n_sample),
            y_range=(self.y_min-0.05, self.y_max)
        )

        p.title.text_font_size = "25px"
        p.title.align = 'center'
        p.yaxis.major_label_text_font_size = "15pt"
        p.yaxis.axis_label_text_font_size = "20pt"

        # Create a source for the rectangles (boxes)
        self.box_source = ColumnDataSource(data=dict(
            left=[(-1*0.05*self.n_sample), int(self.n_sample*0.9)], right=[int(self.n_sample*0.1), self.n_sample],  # X positions for two boxes
            bottom=[0.5*self.y_max, -0.01], top=[self.y_max, 0.2*self.y_max], color=['yellow', 'blue'], name=['High Noise', 'Low Noise']  # Y positions for two boxes
        ))

        quad_renderer = p.quad(left="left", right="right", bottom="bottom", top="top", 
                       source=self.box_source, fill_alpha=0.3, color="color")

        # Add BoxEditTool
        box_edit = BoxEditTool(renderers=[quad_renderer])
        
        p.add_tools(box_edit)

        # Scatter plot
        p.scatter("x", "y", source=self.shared_source, size="size", color="color",
                legend_label="Data", fill_alpha=0.6, line_color='black')

        return p

    def create_image_displays(self):
        return {
            "yellow": Div(text="<h3>High Noise Examples:</h3>",
                           width=400, height=500,
                           stylesheets=[".scroll-box { overflow-y: auto; max-height: 500px; padding: 0px; }"],
                           css_classes=["scroll-box"]
                          ),
            "blue": Div(text="<h3>Low Noise Examples:</h3>",
                         width=400, height=500,
                         stylesheets=[".scroll-box { overflow-y: auto; max-height: 500px; padding: 0px; }"],
                         css_classes=["scroll-box"]
                        )
        }

    def setup_callbacks(self):
        self.step_slider.js_on_change("value", CustomJS(args={"original": self.shared_resource,
                                                              "source": self.shared_source,  
                                                              "image_displays": self.image_displays,
                                                              "box_source": self.box_source,
                                                              "slider": self.step_slider
                                                              },
        code="""
            var step = cb_obj.value;
            var shared_data = original.data;

            source.data["y"] = shared_data["y"][step];
            source.data["x"] = shared_data["x"][step];

            var images = source.data["img"];
            var labels = source.data["label"];
            var x_values = source.data["x"];
            var y_values = source.data["y"];
            var indices = x_values.length;

            var box_data = box_source.data;
            var box_colors = box_data["color"];
            var lefts = box_data["left"];
            var rights = box_data["right"];
            var bottoms = box_data["bottom"];
            var tops = box_data["top"];

            // Create dictionaries to group images by label for each color category
            var images_by_box = {
                "yellow": {},
                "blue": {}
            };

            for (var i = 0; i < indices; i++) {
                var x = x_values[i];
                var y = y_values[i];
                var label = labels[i];
                var imgTag = "<img src='data:image/png;base64," + images[i] + "' width='64' height='64'>";

                for (var j = 0; j < box_colors.length; j++) {
                    if (x >= lefts[j] && x <= rights[j] && y >= bottoms[j] && y <= tops[j]) {
                        var color = box_colors[j];

                        // Initialize label group if not present
                        if (!(label in images_by_box[color])) {
                            images_by_box[color][label] = [];
                        }

                        // Add image to corresponding label group
                        if (images_by_box[color][label].length < 4){
                            images_by_box[color][label].push(imgTag);
                        }
                    }
                }
            }

            // Function to generate HTML for a category
            function generate_html(images_by_label, title) {
                var html = "<h3>" + title + "</h3>";
                
                var sortedLabels = Object.keys(images_by_label).sort();
                sortedLabels.forEach(function(label) {
                    html += "<div style='margin-bottom:10px; max-width: 500;'><b>" + "</b><br>";
                    html += "<div style='display: flex; flex-wrap: wrap; gap: 1px;'>" + images_by_label[label].join("") + "</div></div>";
                });

                return html;
            }

            // Update image display text
            image_displays["yellow"].text = '<div class="scroll-box">' + generate_html(images_by_box["yellow"], "High Noise Examples") + '</div>';
            image_displays["blue"].text = '<div class="scroll-box">' + generate_html(images_by_box["blue"], "Low Noise Examples") + '</div>';

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

        # Trigger selection when data updates
        self.shared_source.js_on_change("data", CustomJS(args={
            "source": self.shared_source, "image_display": self.image_displays,
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
            "source": self.shared_source, "image_display": self.image_displays,
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
        return row(column(self.plot, self.step_slider, self.play_button),
                   row(self.image_displays["yellow"], self.image_displays["blue"]))