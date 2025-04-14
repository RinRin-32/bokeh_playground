from bokeh.models import ColumnDataSource, CustomJS
from bokeh.plotting import figure
from scipy.interpolate import make_interp_spline
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
        self.add_distribution_line()

    def create_plot(self):
        p = figure(height=600, width=600, title="Noise Magnitude", tools="", 
                   x_range=(self.min_x-1.5, self.max_x+1.5), 
                   y_range=(self.min_y, self.max_y))

        p.xaxis.axis_line_color = None
        p.xaxis.major_tick_line_color = None
        p.xaxis.minor_tick_line_color = None

        #p.vbar(x="sig_in", top="noise", source=self.source, width=self.bar_width, color="color", line_color='black')

        return p

    def add_dynamic_y_range(self):
        code = """
        const noise_data = source.data['noise'];
        const alpha_data = source.data['bar_alpha'];
        
        let max_noise = 0;
        for (let i = 0; i < noise_data.length; i++) {
            if (alpha_data[i] > 0) {
                max_noise = Math.max(max_noise, noise_data[i]);
            }
        }

        plot.y_range.end = max_noise * 1.1;
        """
        callback = CustomJS(args=dict(source=self.sync, plot=self.plot), code=code)
        self.sync.js_on_change('data', callback)

    def add_distribution_line(self):
        """Creates a smoothed line over the bar plot to simulate a distribution curve."""
        x = np.array(self.source.data["sig_in"])
        y = np.array(self.source.data["noise"])

        # Only interpolate if there are at least 3 points
        if len(x) >= 3:
            x_sorted_indices = np.argsort(x)
            x_sorted = x[x_sorted_indices]
            y_sorted = y[x_sorted_indices]

            # Smooth interpolation
            x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 200)
            spline = make_interp_spline(x_sorted, y_sorted, k=3)
            y_smooth = spline(x_smooth)

            dist_source = ColumnDataSource(data=dict(x=x_smooth, y=y_smooth))
            self.plot.line(x="x", y="y", source=dist_source, 
                           line_color="orange", line_width=2, line_dash="dashed")

    def get_layout(self):
        return self.plot