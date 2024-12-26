from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.models import Button, Div

class MemoryMapVisualizer:
    def __init__(self, shared_source, colors):
        """
        Initializes the Memory Map Visualizer.
        Args:
        - shared_source: Shared ColumnDataSource for syncing data across modules.
        - colors: List of colors for each class.
        - markers: List of markers for each class.
        """
        self.source = shared_source

        # Set up the plot
        self.plot = figure(title="Memory Map Visualization", width=600, height=600, tools="tap,box_select")
        self.plot.scatter("x", "y", size=8, source=self.source, color="color", marker="marker")

        # Setup selection callback
        self.source.selected.on_change('indices', self.update_selection)

        # Create buttons for confirmation and reset
        self.confirm_button = Button(label="Confirm Selection", button_type="success")
        self.confirm_button.on_click(self.confirm_selection)

        self.reset_button = Button(label="Reset Selection", button_type="warning")
        self.reset_button.on_click(self.reset_selection)

        self.message_div = Div(text="", width=400, height=50)
        self.colors = colors

    def update_selection(self, attr, old, new):
        """Update the selection of points (toggle to red)."""
        selected_indices = self.source.selected.indices
        new_data = self.source.data.copy()

        for idx in range(len(new_data['color'])):
            if idx in selected_indices:
                new_data["color"][idx] = "red"

        self.source.data = new_data

    def confirm_selection(self):
        """Confirm the selected points and lock their colors to grey."""
        new_data = self.source.data.copy()
        selected_indices = self.source.selected.indices

        for idx in selected_indices:
            # Confirm selected points by setting color to grey
            if new_data["color"][idx] == "red":
                new_data["color"][idx] = "grey"

        self.source.data = new_data
        self.source.selected.indices = []  # Clear selection
        self.message_div.text = "Selection confirmed."

    def reset_selection(self):
        """Reset all selections to their original colors."""
        new_data = self.source.data.copy()

        for idx in range(len(new_data["color"])):
            new_data["color"][idx] = self.colors[int(new_data["class"][idx])]

        self.source.data = new_data
        self.source.selected.indices = []  # Clear selection
        self.message_div.text = "Selections reset."

    def get_plot(self):
        """Returns the Bokeh plot object."""
        return self.plot

    def get_layout(self):
        """Returns the Bokeh layout containing the plot and any additional elements."""
        return column(self.get_plot(), self.confirm_button, self.reset_button, self.message_div)