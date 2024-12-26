from bokeh.plotting import figure
from bokeh.layouts import column

class SensitivityVisualizer:
    def __init__(self, shared_source, line_coords):
        """
        Initializes the SensitivityVisualizer with shared data source and optional regression line coordinates.
        
        Args:
        - shared_source: Shared ColumnDataSource for syncing data across modules.
        - line_coords: A dictionary containing 'x_vals' and 'y_vals' for the regression line.
          Example: {'x_vals': [0, 1, 2], 'y_vals': [2, 3, 4]}
        """
        self.source = shared_source
        self.line_coords = line_coords
        self.plot = self.create_plot()

    def create_plot(self):
        """Creates the plot for the scatter data and optional regression line."""
        # Set up the figure
        p = figure(title="Sensitivity Visualization with Regression Line", width=600, height=600)
        p.xaxis.axis_label = 'True Deviation'
        p.yaxis.axis_label = 'Estimated Deviation'

        # Plot all data points using the shared source
        p.scatter(
            x='x', y='y', color='color', marker='marker', size=8, source=self.source
        )

        # Optionally, plot the regression line if provided
        if self.line_coords is not None:
            self.plot_regression_line(p)

        return p

    def plot_regression_line(self, p):
        """Plots the provided regression line on the plot."""
        if 'x_vals' in self.line_coords and 'y_vals' in self.line_coords:
            x_vals = self.line_coords['x_vals']
            y_vals = self.line_coords['y_vals']

            # Plot the regression line with a dotted style
            p.line(x_vals, y_vals, line_width=2, line_color="red", line_dash="dotted", legend_label="Regression Line")
        else:
            raise ValueError("Invalid line coordinates provided. Please provide 'x_vals' and 'y_vals'.")

    def get_plot(self):
        """Returns the Bokeh plot object."""
        return self.plot

    def get_layout(self):
        """Returns the Bokeh layout containing the plot."""
        layout = column(self.get_plot())
        return layout