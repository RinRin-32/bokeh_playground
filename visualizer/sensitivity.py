from bokeh.plotting import figure
from bokeh.layouts import column

class SensitivityVisualizer:
    def __init__(self, shared_source, line_coords):

        self.source = shared_source
        self.line_coords = line_coords
        self.plot = self.create_plot()

    def create_plot(self):
        # Set up the figure
        p = figure(title="Sensitivity Visualization", width=600, height=600)
        p.xaxis.axis_label = 'True Deviation'
        p.yaxis.axis_label = 'Estimated Deviation'

        # Plot all data points using the shared source
        p.scatter(
            x='true_deviation', y='estimated_deviation', color='color', marker='marker', size=8, source=self.source
        )

        # Optionally, plot the regression line if provided
        if self.line_coords is not None:
            self.plot_regression_line(p)

        return p

    def plot_regression_line(self, p):
        if 'x_vals' in self.line_coords and 'y_vals' in self.line_coords:
            x_vals = self.line_coords['x_vals']
            y_vals = self.line_coords['y_vals']

            # Plot the regression line with a dotted style
            p.line(x_vals, y_vals, line_width=2, line_color="red", line_dash="dotted")
        else:
            raise ValueError("Invalid line coordinates provided. Please provide 'x_vals' and 'y_vals'.")

    def get_plot(self):
        return self.plot

    def get_layout(self):
        layout = column(self.get_plot())
        return layout