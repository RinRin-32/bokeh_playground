from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.models import ColumnDataSource

class EvolvingMemoryMapVisualizer:
    def __init__(self, shared_source):
        self.shared_source = shared_source
        self.source = ColumnDataSource(data={"bls": [], "bpe": [], "color": [], "marker": []})
        self.plot = self.create_plot()

    def create_plot(self):
        # Set up the figure
        p = figure(title="Memory Map Visualization",
                   width=600, height=600,
                   tools='tap,box_select,box_zoom,reset,pan')
        p.xaxis.axis_label = 'Bayesian Leverage Score'
        p.yaxis.axis_label = 'Bayesian Prediction Error'

        # Plot the memory map using the source
        p.scatter(
            x='bls', y='bpe', color='color', marker='marker', size=8, source=self.source
        )

        return p

    def update(self, epoch):
        # Get data for the specified epoch
        shared_data = self.shared_source.data
        if epoch in shared_data["epoch"]:
            epoch_index = shared_data["epoch"].index(epoch)

            # Extract relevant data for the epoch
            x = shared_data["bls"][epoch_index]
            y = shared_data["bpe"][epoch_index]

            # Update the source data
            self.source.data = {
                "bls": x,
                "bpe": y,
                "color": shared_data["color"],
                "marker": shared_data['marker']
            }
        else:
            print(f"Epoch {epoch} not found in shared data.")

    def get_plot(self):
        return self.plot

    def get_layout(self):
        return column(self.get_plot())
