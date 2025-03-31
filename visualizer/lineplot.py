from bokeh.models import Button, CustomJS, Slider, ColumnDataSource, Div
from bokeh.layouts import column
import numpy as np
from bokeh.plotting import figure

class LinePlot:
    def __init__(self, shared_source, min_x, max_x):
        self.source = shared_source
        self.min_x = min_x
        self.max_x = max_x

        self.plot = self.create_plot()
        
    def create_plot(self):
        p = figure(height=100,
                           width=600,
                           title="Single Axis Plot",
                           tools="",
                           x_range=(self.min_x-1, self.max_x+1))
        p.yaxis.visible = False
        p.xaxis.axis_line_color = None
        p.xaxis.major_tick_line_color = None
        p.xaxis.minor_tick_line_color = None
        p.scatter("sig_in", "fixed_axis", source=self.source, size='size', color='color',  marker="marker", line_color='black')
        return p

    def get_layout(self):
        """Returns the layout of the plot and controls."""
        return self.plot