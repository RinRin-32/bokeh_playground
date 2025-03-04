from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models import HoverTool, ColumnDataSource, Div, CustomJS, Select
from visualizer.evolvingmpe import EvolvingMemoryMapVisualizer
from collections import defaultdict


class EvolvingLabelNoisePlot:
    def __init__(self, shared_soruce, plot_name):
        self.shared_source = shared_soruce
        self.plot_name = plot_name

        self.plot = self.create_plot()

        self.image_display = Div(
            text="<h3>Selected Images:</h3>", 
            width=300, height=600, 
            stylesheets=[""" .scroll-box { overflow-y: auto; max-height: 600px; padding: 10px; } """], 
            css_classes=["scroll-box"]
        )

        self.setup_callbacks()

    def create_plot(self):
        p = figure(width=800, height=600, tools="reset,save,box_select",
           title=f"{self.plot_name} Label Noise Distribution",
           x_axis_label="Examples", y_axis_label=r"Label Noise ||ε||₂")
        
        p.title.text_font_size = "25px"
        p.title.align = 'center'
        p.xaxis.visible = False
        p.yaxis.major_label_text_font_size = "15pt"
        p.yaxis.axis_label_text_font_size = "20pt"
        p.xaxis.axis_label_text_font_size = "20pt"

        p.scatter("x", "y", source=self.shared_source, size=6, color="color", legend_label="Data", fill_alpha=0.6)

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
        self.shared_source.selected.js_on_change("indices", CustomJS(args={"source": self.shared_source, "image_display": self.image_display}, code="""
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
        return row(self.plot, self.image_display,)