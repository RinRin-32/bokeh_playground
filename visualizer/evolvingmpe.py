from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.models import HoverTool

class EvolvingMemoryMapVisualizer:
    def __init__(self, shared_source, lambda_var_plot=False):
        self.shared_source = shared_source
        self.plot = self.create_plot()

        if lambda_var_plot:
            hover = HoverTool()
            hover.tooltips = [
                ("Average Marginal Vars", "@average_marginal_vars"),
                ("Average Lambda", "@average_lambda")
            ]
            self.plot.add_tools(hover)
        

    def create_plot(self):
        # Set up the figure
        p = figure(title="Memory Map Visualization",
                   width=600, height=600,
                   tools='tap,box_select, box_zoom, reset',
                   active_drag='box_select',
                   )
        p.xaxis.axis_label = 'Bayesian Leverage Score'
        p.yaxis.axis_label = 'Bayesian Prediction Error'

        # Plot the memory map using the source
        p.scatter(x='bls', y='bpe', color='color', marker='marker', alpha='alpha', size='size', source=self.shared_source)

        p.x_range.only_visible = p.y_range.only_visible = True

        return p

    def get_plot(self):
        return self.plot

    def get_layout(self):
        return column(self.get_plot())
