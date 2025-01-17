from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
import numpy as np

class EvolvingSensitivityVisualizer:
    def __init__(self, shared_source):
        self.shared_source = shared_source
        self.source = ColumnDataSource(data={"true_deviation": [], "estimated_deviation": [], "color": [], "marker": []})
        self.plot = self.create_plot()

    def create_plot(self):
        # Set up the figure
        p = figure(title="Sensitivity Visualization",
                   width=600, height=600,
                   tools='tap,box_select,box_zoom,reset,pan')
        p.xaxis.axis_label = 'True Deviation'
        p.yaxis.axis_label = 'Estimated Deviation'

        # Plot all data points using the source
        p.scatter(
            x='true_deviation', y='estimated_deviation', color='color', marker='marker', size=8, source=self.source
        )

        # Initialize the regression line (will be updated dynamically)
        self.regression_line = p.line([], [], line_width=2, line_color="red", line_dash="dotted")

        return p

    def update(self, epoch):
        # Get data for the specified epoch
        shared_data = self.shared_source.data
        if epoch in shared_data["epoch"]:
            epoch_index = shared_data["epoch"].index(epoch)
            estimated_deviation = shared_data["sensitivities"][epoch_index]
            true_deviation = shared_data["softmax_deviations"][epoch_index]

            # Update source data
            self.source.data = {
                "true_deviation": true_deviation,
                "estimated_deviation": estimated_deviation,
                "color": shared_data["color"],
                "marker": shared_data['marker']
            }

            # Update the regression line
            self.update_regression_line()
        else:
            print(f"Epoch {epoch} not found in shared data.")

    def update_regression_line(self):
        # Extract data from the updated source
        true_deviation = self.source.data['true_deviation']
        estimated_deviation = self.source.data['estimated_deviation']

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