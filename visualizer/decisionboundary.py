from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models import Div, Button, ColumnDataSource, GlyphRenderer, Image
import numpy as np

class DecisionBoundaryVisualizer:
    def __init__(self, model, shared_source):
        self.model = model
        self.source = shared_source

        self.X = np.column_stack([self.source.data[feature] for feature in self.source.data if feature not in ['id', 'class', 'color', 'marker', 'estimated_deviation', 'true_deviation', 'bpe', 'bls']])
        self.y = self.source.data['class']

        self.classes = np.unique(self.y) 
        self.message_div = Div(text="", width=400, height=50, styles={"color": "red"})

        self.plot = figure(title="Interactive 2D Classification Visualization", width=600, height=600, tools="tap,box_select")

        self.boundary_x, self.boundary_y = self.calculate_boundaries(self.X, self.y)

        self.plot.scatter("x", "y", size=8, source=self.source, color="color", marker="marker")
        if self.boundary_x is not None and self.boundary_y is not None:
            self.plot.line(x=self.boundary_x, y=self.boundary_y, line_width=2, color="black")
        
        self.source.on_change('data', self.update)
        
    def calculate_boundaries(self, X, y):
        print("Calculating boundaries...")
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            self.message_div.text = "Error: At least two classes are required to fit the model."
            return None, None
        else:
            self.message_div.text = ""
            self.model.fit(X, y)
            
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), 
                                np.linspace(y_min, y_max, 100))
            
            grid = np.c_[xx.ravel(), yy.ravel()]
            zz = self.model.predict_proba(grid)[:, 1]
            zz = zz.reshape(xx.shape)
            
            boundary_mask = np.isclose(zz, 0.7, atol=0.04)
            boundary_x, boundary_y = xx[boundary_mask], yy[boundary_mask]

            return boundary_x, boundary_y

        
    def update(self, attr, old, new):
        mask = np.array(self.source.data["color"]) != "grey"
        x_new = np.array(self.source.data["x"])[mask]
        y_new = np.array(self.source.data["y"])[mask]
        
        X_new = np.vstack((x_new, y_new)).T
        y_new = np.array(self.source.data["class"])[mask].flatten()

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
        
    def get_layout(self):
        return column(self.plot, self.message_div)
