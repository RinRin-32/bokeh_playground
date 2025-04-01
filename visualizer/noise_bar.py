from bokeh.models import ColumnDataSource, CustomJS, Text
from bokeh.plotting import figure
import numpy as np

class BarProjectionPlot:
    def __init__(self, shared_source, sync, min_x, max_x, min_y, max_y, bar_width=0.5):
        self.source = shared_source
        self.bar_width = bar_width
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.sync = sync
        
        self.plot = self.create_plot()
        self.add_dynamic_y_range()

    def create_plot(self):
        p = figure(height=600, width=600, title="Noise Magnitude", tools="", x_range=(self.min_x-1.5, self.max_x+1.5), y_range=(0, self.max_y))
        p.xaxis.axis_line_color = None
        p.yaxis.visible = False
        p.xaxis.major_tick_line_color = None
        p.xaxis.minor_tick_line_color = None

        p.vbar(x="sig_in", top="noise", source=self.source, width=self.bar_width, color="color", line_color='black')
        p.vbar(x="sig_in", top="noise", alpha="bar_alpha", source=self.sync, width=self.bar_width/5, color="color", line_color="color")

        text_source = ColumnDataSource(data=dict(x=[-1], y=[0.5], text=["Noise Magnitude"]))

        return p

    def add_dynamic_y_range(self):
        # CustomJS to adjust y-range based on alpha values
        code = """
        const noise_data = source.data['noise'];
        const alpha_data = source.data['bar_alpha'];
        
        let max_noise = 0;
        for (let i = 0; i < noise_data.length; i++) {
            if (alpha_data[i] > 0) {
                max_noise = Math.max(max_noise, noise_data[i]);
            }
        }

        // Adjust y_range to fit the data with alpha > 0
        plot.y_range.end = max_noise * 1.1;  // Add a small margin
        console.log('using callback');
        """
        callback = CustomJS(args=dict(source=self.sync, plot=self.plot), code=code)
        self.sync.js_on_change('data', callback)

    def get_layout(self):
        """Returns the layout of the plot."""
        return self.plot