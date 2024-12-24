import numpy as np
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, Button, Div, DataTable, TableColumn, Image
from bokeh.layouts import column, row

class DecisionBoundaryVisualizer:
    def __init__(self, model, X, y, classes, colors, markers, n_samples=200):
        """
        Initialize the visualizer with the model and data.
        
        :param model: The classification model (e.g., MLPClassifier).
        :param X: Input features (data points).
        :param y: Target labels.
        :param classes: List of classes (for coloring the plot).
        :param colors: List of colors corresponding to the classes.
        :param n_samples: Number of samples to use in the plot (default is 200).
        """
        self.model = model
        self.X = X
        self.y = y
        self.classes = classes
        self.colors = colors
        self.n_samples = n_samples

        # Create ColumnDataSource for points
        self.source = ColumnDataSource(data={
            "x": X[:, 0],
            "y": X[:, 1],
            "class": y,
            "color": [colors[int(cls)] for cls in y],
            "prev": [colors[int(cls)] for cls in y],
            "marker": [markers[int(cls)] for cls in y]
        })
        self.selected_source = ColumnDataSource(data={"x": [], "y": [], "class": [], "status": []})

        # Initialize buttons
        self.confirm_button = Button(label="Confirm Selection", width=200)
        self.reset_button = Button(label="Reset", width=200)
        self.message_div = Div(text="", width=400, height=50, styles={"color": "red"})

        # Initial plot
        self.plot = figure(title="Interactive 2D Classification Visualization", width=600, height=600, tools="tap,box_select,lasso_select")

        # Set up initial decision boundary plot
        self.xx, self.yy, self.Z = self.calculate_boundaries(self.X, self.y)

        # Create plot layout
        self.plot.scatter("x", "y", size=8, source=self.source, color="color", marker="marker")
        if self.Z is not None:
            self.plot.image(image=[self.Z], x=self.xx.min(), y=self.yy.min(), dw=self.xx.max()-self.xx.min(),
                            dh=self.yy.max()-self.yy.min(), palette=["blue", "red"], alpha=0.3)

        # Add selection functionality
        self.ind = []
        self.setup_callbacks()

    def calculate_boundaries(self, X_new, y_new):
        """
        Calculate decision boundaries using the model.
        """
        # Fit the model
        unique_classes = np.unique(y_new)
        if len(unique_classes) < 2:
            self.message_div.text = "Error: At least two classes are required to fit the model."
            return None, None, None
        else:
            self.message_div.text = ""  # Clear the message
            self.model.fit(X_new, y_new)
            x_min, x_max = X_new[:, 0].min() - 1, X_new[:, 0].max() + 1
            y_min, y_max = X_new[:, 1].min() - 1, X_new[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                 np.linspace(y_min, y_max, 100))
            Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
            return xx, yy, Z.reshape(xx.shape)

    def update_selection(self, attr, old, new):
        """
        Update the selection of points based on user interaction.
        """
        selected_indices = self.source.selected.indices
        remaining_indices = []
        temp_data = {"x": [], "y": [], "class": [], "status": []}
        new_data = self.source.data.copy()

        for idx in selected_indices:
            if len(selected_indices) > 1:
                if new_data["color"][idx] != "red":
                    temp_data["x"].append(new_data["x"][idx])
                    temp_data["y"].append(new_data["y"][idx])
                    temp_data["class"].append(new_data["class"][idx])
                    if new_data["color"][idx] != 'grey':
                        temp_data["status"].append('removing')
                        new_data["prev"][idx] = new_data["color"][idx]
                        new_data["color"][idx] = "red"
                    else:
                        temp_data["status"].append('adding')
                        new_data["prev"][idx] = new_data["color"][idx]
                        new_data["color"][idx] = "lime"
                remaining_indices.append(idx)
            else:
                if new_data["color"][idx] == 'red':
                    new_data["color"][idx] = new_data["prev"][idx]
                else:
                    temp_data["x"].append(new_data["x"][idx])
                    temp_data["y"].append(new_data["y"][idx])
                    temp_data["class"].append(new_data["class"][idx])
                    if new_data["color"][idx] != 'grey':
                        temp_data["status"].append('removing')
                        new_data["prev"][idx] = new_data["color"][idx]
                        new_data["color"][idx] = "red"
                    else:
                        temp_data["status"].append('adding')
                        new_data["prev"][idx] = new_data["color"][idx]
                        new_data["color"][idx] = "lime"

        self.source.data = new_data
        self.selected_source.stream(temp_data)
        if len(selected_indices) > 1:
            self.ind.extend(remaining_indices)
        else:
            self.ind.extend(selected_indices)

    def confirm_selection(self):
        """
        Confirm the selected points and update the model and decision boundaries.
        """
        selected_indices = self.ind
        new_data = self.source.data.copy()

        for idx in selected_indices:
            if new_data["prev"][idx] != "grey" and (new_data["color"][idx] != new_data["prev"][idx]):
                new_data["color"][idx] = "grey"
            elif new_data["class"][idx] == 0:
                new_data["prev"][idx] =  new_data["color"][idx]
                new_data["color"][idx] = "blue"
            else:
                new_data["prev"][idx] =  new_data["color"][idx]
                new_data["color"][idx] = "green"
        self.source.data = new_data

        # Recalculate decision boundaries
        mask = np.array(new_data["color"]) != "grey"
        X_new = np.column_stack((np.array(new_data["x"])[mask], np.array(new_data["y"])[mask]))
        y_new = np.array(new_data["class"])[mask]

        xx, yy, Z = self.calculate_boundaries(X_new, y_new)
        if Z is not None:
            self.plot.renderers = [r for r in self.plot.renderers if not isinstance(r, Image)]
            if len(self.plot.renderers) > 2:
                self.plot.renderers.remove(self.plot.renderers[-1])
            self.plot.image(image=[Z], x=xx.min(), y=yy.min(), dw=xx.max()-xx.min(),
                            dh=yy.max()-yy.min(), palette=["blue", "red"], alpha=0.3)

        # Clear the selection
        self.source.selected.indices = []
        self.selected_source.data = {"x": [], "y": [], "class": [], "status": []}
        self.ind.clear()

    def reset_selection(self):
        """
        Reset all selections and recalculate the decision boundaries.
        """
        new_data = self.source.data.copy()
        for i in range(len(new_data["color"])):
            new_data["color"][i] = self.colors[int(new_data["class"][i])]
        self.source.data = new_data

        # Recalculate decision boundaries
        xx, yy, Z = self.calculate_boundaries(self.X, self.y)
        if Z is not None:
            self.plot.renderers = [r for r in self.plot.renderers if not isinstance(r, Image)]
            self.plot.renderers = self.plot.renderers[0:2]
            self.plot.image(image=[Z], x=xx.min(), y=yy.min(), dw=xx.max()-xx.min(),
                            dh=yy.max()-yy.min(), palette=["blue", "red"], alpha=0.3)

        # Clear the selection
        self.source.selected.indices = []
        self.selected_source.data = {"x": [], "y": [], "class": [], "status": []}
        self.ind.clear()

    def setup_callbacks(self):
        """
        Set up all necessary callbacks for interactions.
        """
        self.source.selected.on_change("indices", self.update_selection)
        self.confirm_button.on_click(self.confirm_selection)
        self.reset_button.on_click(self.reset_selection)

    def layout(self):
        """
        Return the layout for the Bokeh server.
        """
        return column(self.plot, row(self.confirm_button, self.reset_button), self.message_div)