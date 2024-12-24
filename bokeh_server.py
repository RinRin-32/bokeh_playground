import numpy as np
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, TapTool, LassoSelectTool, DataTable, TableColumn, Button, Div
from bokeh.layouts import column, row
from sklearn.linear_model import LogisticRegression
from bokeh.models import Image
from sklearn.neural_network import MLPClassifier

# Generate random 2D data
np.random.seed(42)
n_samples = 200

# Class 0
x0 = np.random.normal(loc=5.0, scale=3.0, size=(n_samples, 2))
y0 = np.zeros(n_samples)

# Class 1
x1 = np.random.normal(loc=-5.0, scale=4.5, size=(n_samples, 2))
y1 = np.ones(n_samples)

# Combine data
X = np.vstack((x0, x1))
y = np.hstack((y0, y1))
model = MLPClassifier(hidden_layer_sizes=(50,50), max_iter=300, random_state=42)

# Create a ColumnDataSource
source = ColumnDataSource(data={
    "x": X[:, 0],
    "y": X[:, 1],
    "class": y,
    "color": ["blue" if cls == 0 else "green" for cls in y],
    "prev": ["blue" if cls == 0 else "green" for cls in y],
})
selected_source = ColumnDataSource(data={"x": [], "y": [], "class": [], "status": []})
ind = []

# Add a Div for messages
message_div = Div(text="", width=400, height=50, styles={"color": "red"})

# Function to calculate decision boundaries
def calculate_boundaries(X, y, model):
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        reset_selection()
        message_div.text = "Error: At least two classes are required to fit the model."
        return None, None, None
    else:
        message_div.text = ""  # Clear the message
        model.fit(X, y)
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        return xx, yy, Z.reshape(xx.shape)

# Initial calculation
xx, yy, Z = calculate_boundaries(X, y, model)

# Create Bokeh figure
p = figure(title="Interactive Logistic Regression", width=600, height=600, tools="tap,box_select,lasso_select")
if Z is not None:
    p.image(image=[Z], x=xx.min(), y=yy.min(), dw=xx.max()-xx.min(),
            dh=yy.max()-yy.min(), palette=["blue", "red"], alpha=0.3)

# Add scatter points
p.scatter("x", "y", size=8, source=source, color="color")

# Table for previewing selected points
columns = [
    TableColumn(field="class", title="Class"),
    TableColumn(field="status", title="Status")
]
data_table = DataTable(source=selected_source, columns=columns, width=400, height=200)

# Button to confirm selection
confirm_button = Button(label="Confirm Selection", width=200)

# Button to reset all selections
reset_button = Button(label="Reset", width=200)

# Callback for updating the selection stack
def update_selection(attr, old, new):
    selected_indices = source.selected.indices
    remaining_indices = []
    temp_data = {"x": [], "y": [], "class": [], "status": []}
    new_data = source.data.copy()
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
                    new_data["color"][idx] = " lime"
    source.data = new_data
    selected_source.stream(temp_data)
    if len(selected_indices) > 1:
        ind.extend(remaining_indices)
    else:
        ind.extend(selected_indices)

# Callback for confirming selection
def confirm_selection():
    selected_indices = ind
    new_data = source.data.copy()

    for idx in selected_indices:
        if new_data["prev"][idx] != "grey" and (new_data["color"][idx] != new_data["prev"][idx]):
            new_data["color"][idx] = "grey"
        elif new_data["class"][idx] == 0:
            new_data["prev"][idx] =  new_data["color"][idx]
            new_data["color"][idx] = "blue"
        else:
            new_data["prev"][idx] =  new_data["color"][idx]
            new_data["color"][idx] = "green"
    source.data = new_data

    # Recalculate decision boundaries
    mask = np.array(new_data["color"]) != "grey"
    X_new = np.column_stack((np.array(new_data["x"])[mask], np.array(new_data["y"])[mask]))
    y_new = np.array(new_data["class"])[mask]

    xx, yy, Z = calculate_boundaries(X_new, y_new, model)
    if Z is not None:
        p.renderers = [r for r in p.renderers if not isinstance(r, Image)]
        if len(p.renderers) > 2:
            p.renderers.remove(p.renderers[-1])
        p.image(image=[Z], x=xx.min(), y=yy.min(), dw=xx.max()-xx.min(),
                dh=yy.max()-yy.min(), palette=["blue", "red"], alpha=0.3)

    # Clear the selection and stack
    source.selected.indices = []
    selected_source.data = {"x": [], "y": [], "class": [], "status": []}
    ind.clear()

# Callback for resetting all selections
def reset_selection():
    # Reset source data colors to original
    new_data = source.data.copy()
    for i in range(len(new_data["color"])):
        new_data["color"][i] = "blue" if new_data["class"][i] == 0 else "green"
    source.data = new_data

    # Recalculate decision boundaries using all data
    xx, yy, Z = calculate_boundaries(X, y, model)
    if Z is not None:
        p.renderers = [r for r in p.renderers if not isinstance(r, Image)]
        p.renderers = p.renderers[0:2]
        p.image(image=[Z], x=xx.min(), y=yy.min(), dw=xx.max()-xx.min(),
                dh=yy.max()-yy.min(), palette=["blue", "red"], alpha=0.3)

    # Clear the selection and stack
    source.selected.indices = []
    selected_source.data = {"x": [], "y": [], "class": [], "status": []}
    ind.clear()

# Attach callbacks
source.selected.on_change("indices", update_selection)
confirm_button.on_click(confirm_selection)
reset_button.on_click(reset_selection)

# Layout and show
layout = column(p, row(confirm_button, reset_button), message_div)
curdoc().add_root(layout)