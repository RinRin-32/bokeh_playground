import numpy as np
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, TapTool, DataTable, TableColumn, Button
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
source = ColumnDataSource(data={
    "x": X[:, 0],
    "y": X[:, 1],
    "class": y,
    "color": ["blue" if cls == 0 else "green" for cls in y]
})
selected_source = ColumnDataSource(data={"x": [], "y": [], "class": [], "status": []})
ind = []

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

# Table for previewing selected points
columns = [
    TableColumn(field="x", title="X"),
    TableColumn(field="y", title="Y"),
    TableColumn(field="class", title="Class"),
    TableColumn(field="status", title="Status")
]
data_table = DataTable(source=selected_source, columns=columns, width=400, height=200)

# Button to confirm selection
confirm_button = Button(label="Confirm Selection", width=200)

# Callback for updating the selection stack
def update_selection(attr, old, new):
    selected_indices = source.selected.indices
    temp_data = {"x": [], "y": [], "class": [], "status": []}
    new_data = source.data.copy()
    for idx in selected_indices:
        temp_data["x"].append(new_data["x"][idx])
        temp_data["y"].append(new_data["y"][idx])
        temp_data["class"].append(new_data["class"][idx])
        if new_data["color"][idx] != 'grey':
            temp_data["status"].append('removing')
        else:
            temp_data["status"].append('adding')
        new_data["color"][idx] = "red"
    source.data = new_data
    selected_source.stream(temp_data)
    ind.extend(selected_indices)

# Callback for confirming selection
def confirm_selection():
    selected_indices = ind
    new_data = source.data.copy()

    for idx in selected_indices:
        if new_data["color"][idx] != "grey":
            new_data["color"][idx] = "grey"
        elif new_data["class"][idx] == 0:
            new_data["color"][idx] = "blue"
        else:
            new_data["color"][idx] = "green"
    source.data = new_data

    # Recalculate decision boundaries
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

    # Clear the selection and stack
    source.selected.indices = []
    selected_source.data = {"x": [], "y": [], "class": [], "status": []}

# Attach callbacks
source.selected.on_change("indices", update_selection)
confirm_button.on_click(confirm_selection)

# Layout and show
layout = column(p, data_table, confirm_button)
curdoc().add_root(layout)
