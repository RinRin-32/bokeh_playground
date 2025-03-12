from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models import HoverTool, ColumnDataSource, Div, CustomJS, Select
from visualizer.evolvingmpe import EvolvingMemoryMapVisualizer
from collections import defaultdict


class EvolvingLabelNoisePlot:
    def __init__(self, shared_soruce, plot_name, y_range, n_sample):
        self.shared_source = shared_soruce
        self.plot_name = plot_name
        self.n_sample = n_sample

        self.y_min = y_range[0]
        self.y_max = y_range[1]

        self.plot = self.create_plot()

        self.image_display = Div(
            text="<h3>Selected Images:</h3>", 
            sizing_mode="stretch_width", height=500, 
            stylesheets=[""" .scroll-box { overflow-y: auto; max-height: 600px; padding: 10px; } """], 
            css_classes=["scroll-box"]
        )

        self.setup_callbacks()

    def create_plot(self):
        p = figure(
            width=800, height=500,
            tools="lasso_select, reset,save,box_select",
            active_drag='box_select',
            title=f"{self.plot_name} Label Noise Distribution",
            x_axis_label="Sorted Sample By Noise", y_axis_label=r"Label Noise ||ε||₂",
            x_range=(0, self.n_sample),
            y_range=(self.y_min, self.y_max))
        
        p.title.text_font_size = "25px"
        p.title.align = 'center'
        p.xaxis.visible = False
        p.yaxis.major_label_text_font_size = "15pt"
        p.yaxis.axis_label_text_font_size = "20pt"
        p.xaxis.axis_label_text_font_size = "20pt"
        p.xaxis.visible=True
        p.xaxis.major_label_text_font_size = "0pt"
        p.min_border_bottom = 50

        p.scatter("x", "y", source=self.shared_source, size=6, color="color", legend_label="Data", fill_alpha=0.6)

        hover = HoverTool(tooltips="""
                <div>
                    <img src="data:image/png;base64,@img" width="28" height="28"></img>
                    <br>
                    <b>Label:</b> @label
                    <br>
                    <img src="data:image/png;base64,@noise_chart" width="150" height="100"></img>
                </div>
        """)
        p.add_tools(hover)

        return p
    
    def setup_callbacks(self):
        self.shared_source.selected.js_on_change("indices", CustomJS(args={"source": self.shared_source, "image_display": self.image_display}, code="""
            var indices = source.selected.indices;
            var images = source.data["img"];
            var labels = source.data["label"];
            
            var groupedImages = {};
            
            // Collect images by label
            for (var i = 0; i < indices.length; i++) {
                var label = labels[indices[i]];
                var imgTag = "<img src='data:image/png;base64," + images[indices[i]] + "' width='64' height='64'>";
                
                if (!(label in groupedImages)) {
                    groupedImages[label] = [];
                }
                groupedImages[label].push(imgTag);
            }
            
            // Construct the HTML grouped by label
            var html = "<h3>Selected Images:</h3>";
            var sortedLabels = Object.keys(groupedImages).sort();
            
            sortedLabels.forEach(function(label) {
                html += "<div style='margin-bottom:10px;'><b>Label: " + label + "</b><br>";
                html += groupedImages[label].join(" ") + "</div>";
            });

            image_display.text = '<div class="scroll-box">' + html + '</div>';
        """))
    
    def get_layout(self):
        return row(self.plot, self.image_display,)