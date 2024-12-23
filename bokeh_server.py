import numpy as np
from bokeh.plotting import figure, show, curdoc
from bokeh.models import ColumnDataSource, TapTool
from bokeh.layouts import column
from sklearn.linear_model import LogisticRegression
from bokeh.models import Image

# Generate random 2D data
np.random.seed(42)
n_samples = 100

# Class 0
x0 = np.random.normal(loc=2.0, scale=1.0, size=(n_samples, 2))
y0 = np.zeros(n_samples)

# Class 1
x1 = np.random.normal(loc=-2.0, scale=1.0, size=(n_samples, 2))
y1 = np.ones(n_samples)

# Combine data
X = np.vstack((x0, x1))
y = np.hstack((y0, y1))

# Create a ColumnDataSource
source = ColumnDataSource(data={"x": X[:, 0], "y": X[:, 1], "class": y, "color": ["blue"]*n_samples*2})

# Function to calculate decision boundaries
def calculate_boundaries(X, y):
    if len(X) > 1:
        model = LogisticRegression()
        model.fit(X, y)
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        return xx, yy, Z.reshape(xx.shape)
    return None, None, None

# Initial calculation
xx, yy, Z = calculate_boundaries(X, y)

# Create Bokeh figure
p = figure(title="Interactive Logistic Regression", width=600, height=600, tools="tap")
if Z is not None:
    p.image(image=[Z], x=xx.min(), y=yy.min(), dw=xx.max()-xx.min(),
            dh=yy.max()-yy.min(), palette=["blue", "red"], alpha=0.3)

# Add scatter points
p.scatter("x", "y", size=8, source=source, color="color")

# Callback function to remove a selected point
def remove_point(attr, old, new):
    selected_index = source.selected.indices
    if selected_index:
        #new_data = {key: np.delete(value, selected_index, axis=0) for key, value in source.data.items()}
        new_data = source.data.copy()
        for i in selected_index:
            if new_data["color"][i] != 'grey':
                new_data["color"][i] = 'grey'
            else:
                new_data["color"][i] = 'blue'

        source.data = new_data


        mask = np.array(new_data["color"]) != "grey"
        X_new = np.column_stack((np.array(new_data["x"])[mask], np.array(new_data["y"])[mask]))
        y_new = np.array(new_data["class"])[mask]

        xx, yy, Z = calculate_boundaries(X_new, y_new)
        if Z is not None:
            p.renderers = [r for r in p.renderers if not isinstance(r, Image)]
            if len(p.renderers) > 2:
                p.renderers.remove(p.renderers[-1])
            p.image(image=[Z], x=xx.min(), y=yy.min(), dw=xx.max()-xx.min(),
                    dh=yy.max()-yy.min(), palette=["blue", "red"], alpha=0.3)

source.selected.on_change("indices", remove_point)

# Layout and show
layout = column(p)
curdoc().add_root(layout)
show(layout)
