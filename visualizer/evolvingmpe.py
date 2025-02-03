from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.models import ColumnDataSource

class EvolvingMemoryMapVisualizer:
    def __init__(self, shared_source):
        self.shared_source = shared_source
        self.plot = self.create_plot()

    def create_plot(self):
        # Set up the figure
        p = figure(title="Memory Map Visualization",
                   width=600, height=600,
                   tools='tap,box_select,reset')
        p.xaxis.axis_label = 'Bayesian Leverage Score'
        p.yaxis.axis_label = 'Bayesian Prediction Error'

        # Plot the memory map using the source
        p.scatter(x='bls', y='bpe', color='color', marker='marker', alpha='alpha', size='size', source=self.shared_source)

        return p

    def get_plot(self):
        return self.plot

    def get_layout(self):
        return column(self.get_plot())
