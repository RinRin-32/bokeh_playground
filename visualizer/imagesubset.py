from bokeh.models import ColumnDataSource, HoverTool, Div, CustomJS
from bokeh.layouts import column, gridplot
from bokeh.plotting import figure, curdoc

class ImageSet:
    def __init__(self, sources, subsample_image):
        self.n_samples = len(sources)  # Number of sampled datapoints
        self.sources = sources

        # Convert base64 images into HTML <img> tags
        self.image_divs = [
            Div(text=f'<img src="data:image/png;base64,{subsample_image[i]}" width="50" height="50">')
            for i in range(self.n_samples)
        ]

        # Create bar charts
        self.plot = self.create_plot()

    def create_plot(self):
        combined = [
            column(self.image_divs[i], self.create_bar_chart(self.sources[i], f"Class {i}"))
            for i in range(self.n_samples)
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