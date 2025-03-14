from bokeh.models import ColumnDataSource, HoverTool, Div
from bokeh.layouts import column, gridplot
from bokeh.plotting import figure
import numpy as np

class ImageSet:
    def __init__(self, subsample_noise_epoch, subsample_image):
        self.noise_data = np.array(subsample_noise_epoch[0])  # Extract first epoch

        # Create a ColumnDataSource for each sampled datapoint
        self.sources = [
            ColumnDataSource(data={"categories": [str(i) for i in range(10)], "values": self.noise_data[i]})
            for i in range(len(self.noise_data))
        ]

        # Convert base64 images into HTML <img> tags
        self.image_divs = [
            Div(text=f'<img src="data:image/png;base64,{subsample_image[i]}" width="50" height="50">')
            for i in range(len(subsample_image))
        ]

        # Create bar charts for each sampled datapoint
        self.plot = self.create_plot()

    def create_plot(self):
        # Combine images with charts
        combined = [
            column(self.image_divs[i], self.create_bar_chart(self.sources[i], f"Sample {i}"))
            for i in range(len(self.sources))
        ]
        return gridplot([combined[i:i+5] for i in range(0, len(combined), 5)])  # 5 per row

    def create_bar_chart(self, source, title="Bar Chart"):
        p = figure(x_range=[str(i) for i in range(10)], height=100, width=100, title=title, tools="")
        p.vbar(x="categories", top="values", width=0.5, source=source, color="navy")

        # Add hover tool
        hover = HoverTool(tooltips=[("Class", "@categories"), ("Noise", "@values")])
        p.add_tools(hover)

        p.xgrid.grid_line_color = None
        p.y_range.start = 0

        return p

    def get_layout(self):
        return column(self.plot)