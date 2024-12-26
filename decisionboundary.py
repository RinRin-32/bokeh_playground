import numpy as np
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, Button, Div, Image, GlyphRenderer
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
        self.boundary_x, self.boundary_y= self.calculate_boundaries(self.X, self.y)

        # Create plot layout
        self.plot.scatter("x", "y", size=8, source=self.source, color="color", marker="marker")
        if self.boundary_x is not None and self.boundary_y is not None:
            self.plot.line(x=self.boundary_x, y=self.boundary_y, line_width=2, color="black")

        # Add selection functionality
        self.ind = []
        self.setup_callbacks()

    def calculate_boundaries(self, X, y):
        """
        Calculate decision boundaries using the model.
        """
        # Fit the model
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            self.reset_selection()
            self.message_div.text = "Error: At least two classes are required to fit the model."
            return None, None
        else:
            self.message_div.text = ""  # Clear the message
            self.model.fit(X, y)
            
            # Create a grid of points
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), 
                                np.linspace(y_min, y_max, 100))
            
            # Flatten the grid for predictions
            grid = np.c_[xx.ravel(), yy.ravel()]
            zz = self.model.predict_proba(grid)[:, 1]
            zz = zz.reshape(xx.shape)
            
            # Find the decision boundary
            boundary_mask = np.isclose(zz, 0.7, atol=0.04)
            boundary_x, boundary_y = xx[boundary_mask], yy[boundary_mask]
            
            return boundary_x, boundary_y


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

        self.boundary_x, self.boundary_y = self.calculate_boundaries(X_new, y_new)
        if self.boundary_x is not None and self.boundary_y is not None:

            for renderer in self.plot.renderers:
                if isinstance(renderer, (GlyphRenderer, Image)):
                    renderer.glyph.line_alpha = 0.3  # Reduce line visibility
                    # For some reason, on first launch, the line wouldn't reduce visibility but launching the server after that solves it?

            self.plot.renderers = [r for r in self.plot.renderers if not isinstance(r, Image)]
            if len(self.plot.renderers) > 2:
                self.plot.renderers.remove(self.plot.renderers[-1])
            
            self.plot.line(x=self.boundary_x, y=self.boundary_y, line_width=2, color="black")

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
        self.boundary_x, self.boundary_y = self.calculate_boundaries(self.X, self.y)
        if self.boundary_x is not None and self.boundary_y is not None:
            self.plot.renderers = [r for r in self.plot.renderers if not isinstance(r, Image)]
            self.plot.renderers = self.plot.renderers[0:2]
            self.plot.line(x=self.boundary_x, y=self.boundary_y, line_width=2, color="black")

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