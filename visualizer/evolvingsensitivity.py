from bokeh.plotting import figure
from bokeh.layouts import column
import numpy as np

class EvolvingSensitivityVisualizer:
    def __init__(self, shared_source):
        self.shared_source = shared_source
        self.plot = self.create_plot()

    def create_plot(self):
        # Set up the figure
        p = figure(title="Sensitivity Visualization",
                   width=600, height=600,
                   tools='tap,box_select,reset')
        p.xaxis.axis_label = 'True Deviation'
        p.yaxis.axis_label = 'Estimated Deviation'

        # Plot all data points using the source
        p.scatter(
            x='softmax_deviations', y='sensitivities', color='color', marker='marker', size=8, source=self.shared_source
        )

        # Initialize the regression line (will be updated dynamically)
        self.regression_line = p.line([], [], line_width=2, line_color="red", line_dash="dotted")

        return p

    def update(self):
        self.update_regression_line()

    def update_regression_line(self):
        # Extract data from the updated source
        true_deviation = self.shared_source.data['softmax_deviations']
        estimated_deviation = self.shared_source.data['sensitivities']

        if len(true_deviation) > 1:  # Ensure there's enough data for regression
            # Calculate the linear regression coefficients
            slope, intercept = np.polyfit(true_deviation, estimated_deviation, 1)

            # Generate x and y values for the regression line
            x_vals = np.linspace(min(true_deviation), max(true_deviation), 100)
            y_vals = slope * x_vals + intercept

            # Update the regression line data
            self.regression_line.data_source.data = {"x": x_vals, "y": y_vals}

    def get_plot(self):
        return self.plot

    def get_layout(self):
        return column(self.get_plot())