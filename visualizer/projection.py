from bokeh.models import Button, CustomJS, Slider, ColumnDataSource, Div
from bokeh.layouts import column
import numpy as np
from bokeh.plotting import figure

class ProjectionPlot:
    def __init__(self, shared_source, min_x, max_x):
        self.source = shared_source
        self.min = min_x
        self.max = max_x

        self.plot = self.create_plot()
        
    def create_plot(self):
        p = figure(height=600,
                           width=600,
                           title="Sigmoid Plot",
                           tools="")
        p.yaxis.visible = False
        p.xaxis.axis_line_color = None
        p.xaxis.major_tick_line_color = None
        p.xaxis.minor_tick_line_color = None

        x_sigmoid = np.linspace(self.min+0.1, self.max-0.1, 100)
        sigmoid_y = 1 / (1 + np.exp(-x_sigmoid))
        p.line(x_sigmoid, sigmoid_y, color="blue", line_width=2, legend_label="Sigmoid")

        p.scatter("sig_in", "logits", source=self.source, size='size', color='color',  marker="marker", line_color='black')
        p.legend.location = "top_left"

        return p

    def get_layout(self):
        """Returns the layout of the plot and controls."""
        return self.plot