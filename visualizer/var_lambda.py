from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.models import HoverTool

class VarianceLambdaPlot:
    def __init__(self, shared_source):
        self.shared_source = shared_source
        self.plot = self.create_plot()

        hover = HoverTool()
        hover.tooltips = [
            ("Average Marginal Vars", "@average_marginal_vars"),
            ("Average Lambda", "@average_lambda")
        ]
        self.plot.add_tools(hover)

    def create_plot(self):
        # Set up the figure
        p = figure(title="Variance vs Lambda Plot",
                   width=600, height=600,
                   tools='tap,box_select, box_zoom, reset',
                   active_drag='box_select')
        p.xaxis.axis_label = 'Marginal Variance'
        p.yaxis.axis_label = 'Lambda'

        # Plot the memory map using the source
        p.scatter(x='average_marginal_vars', y='average_lambda', color='color', marker='marker', alpha='alpha', size='size', source=self.shared_source)

        return p

    def get_plot(self):
        return self.plot

    def get_layout(self):
        return column(self.get_plot())
