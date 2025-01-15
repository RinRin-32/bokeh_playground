from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.models import Button, Div

class MemoryMapVisualizer:
    def __init__(self, shared_source, colors, decisionboundaryvisualizer):
        self.source = shared_source

        # Set up the plot
        self.plot = figure(title="Memory Map Visualization",
                           width=600, height=600,
                           tools="tap,box_select,box_zoom,reset,pan",
                           active_drag="box_select")
        self.plot.xaxis.axis_label = 'BLS'  # Label for x-axis
        self.plot.yaxis.axis_label = 'BPE'
        self.plot.scatter("bls", "bpe", size=8, source=self.source, color="color", marker="marker")

        self.decisionboundaryvisualizer = decisionboundaryvisualizer

        # Setup selection callback
        self.source.selected.on_change('indices', self.update_selection)

        # Create buttons for confirmation and reset
        self.confirm_button = Button(label="Confirm Selection", button_type="success")
        self.confirm_button.on_click(self.confirm_selection)

        self.reset_button = Button(label="Reset Selection", button_type="danger")
        self.reset_button.on_click(self.reset_selection)

        self.inverse_button = Button(label="Invert Selection", button_type="primary")
        self.inverse_button.on_click(self.invert_selection)

        self.message_div = Div(text="", width=400, height=25)
        self.colors = colors
        self.ind = []

    def update_selection(self, attr, old, new):
        selected_indices = self.source.selected.indices
        new_data = self.source.data.copy()

        for idx in range(len(new_data['color'])):
            if idx in selected_indices:
                if new_data["color"][idx] != "red":
                    new_data["color"][idx] = "red"
                else:
                    new_data["color"][idx] = self.colors[int(new_data["class"][idx])]

        self.source.data = new_data
        self.ind.extend(selected_indices)

    def confirm_selection(self):
        new_data = self.source.data.copy()

        for idx in self.ind:
            # Confirm selected points by setting color to grey
            if new_data["color"][idx] == "red":
                new_data["color"][idx] = "grey"

        self.source.data = new_data
        self.source.selected.indices = []  # Clear selection
        self.message_div.text = "Selection confirmed."
        self.decisionboundaryvisualizer.update(None, None, None)

    def reset_selection(self):
        new_data = self.source.data.copy()

        for idx in range(len(new_data["color"])):
            new_data["color"][idx] = self.colors[int(new_data["class"][idx])]

        self.source.data = new_data
        self.source.selected.indices = []  # Clear selection
        self.message_div.text = "Selections reset."
        self.decisionboundaryvisualizer.update(None, None, None)

    def invert_selection(self):
        new_data = self.source.data.copy()
        count = 0

        for idx in range(len(new_data["color"])):
            if new_data["color"][idx] != 'grey':
                new_data["color"][idx] = 'grey'
            elif new_data["color"][idx] == 'red':
                pass
            else:
                if new_data["color"][idx] != self.colors[int(new_data["class"][idx])]:
                    new_data["color"][idx] = self.colors[int(new_data["class"][idx])]
                else:
                    count+=1
        if count < len(new_data["color"]):
            print("inverted")
            self.source.data = new_data
            self.decisionboundaryvisualizer.update(None, None, None)
        else:
            self.reset_selection
        self.source.selected.indices = []

    def get_plot(self):
        return self.plot

    def get_layout(self):
        return column(self.get_plot(), self.confirm_button, self.reset_button, self.message_div, self.inverse_button)