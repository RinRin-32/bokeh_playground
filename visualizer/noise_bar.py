from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
import numpy as np

class BarProjectionPlot:
    def __init__(self, shared_source, bar_width=0.001):
        self.source = shared_source
        self.bar_width = bar_width
        
        self.plot = self.create_plot()
        
    def create_plot(self):
        p = figure(height=600, width=600, title="Noise Distribution", tools="")
        p.yaxis.visible = False
        p.xaxis.axis_line_color = None
        p.xaxis.major_tick_line_color = None
        p.xaxis.minor_tick_line_color = None
        
        # Add vertical bars instead of scatter points
        p.vbar(x="sig_in", top="noise", source=self.source, width=self.bar_width, color="color", line_color='black')
        
        return p
    
    def get_layout(self):
        """Returns the layout of the plot."""
        return self.plot
