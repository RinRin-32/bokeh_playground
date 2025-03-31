from bokeh.plotting import figure
from bokeh.layouts import column
import numpy as np

class SensitivityVisualizer:
    def __init__(self, shared_source):
        self.source = shared_source
        self.plot = self.create_plot()

    def create_plot(self):
        # Set up the figure
        p = figure(title="Sensitivity Visualization",
                   width=600, height=600,
                   tools='tap,box_select,box_zoom,reset,pan',
                   active_drag='box_select')
        p.xaxis.axis_label = 'True Deviation'
        p.yaxis.axis_label = 'Estimated Deviation'

        # Plot all data points using the shared source
        p.scatter(
            x='true_deviation', y='estimated_deviation', color='color', marker='marker', size=8, source=self.source
        )

        # Plot the regression line based on the scatter plot data
        self.plot_regression_line(p)

        return p

    def plot_regression_line(self, p):
        # Extract data from the shared source
        true_deviation = self.source.data['true_deviation']
        estimated_deviation = self.source.data['estimated_deviation']

        # Calculate the linear regression coefficients
        slope, intercept = np.polyfit(true_deviation, estimated_deviation, 1)

        # Generate x and y values for the regression line
        x_vals = np.linspace(min(true_deviation), max(true_deviation), 100)
        y_vals = slope * x_vals + intercept

        # Plot the regression line with a dotted style
        p.line(x_vals, y_vals, line_width=2, line_color="red", line_dash="dotted")

    def get_plot(self):
        return self.plot

    def get_layout(self):
        layout = column(self.get_plot())
        return layout