from bokeh.models import Button, CustomJS, Slider, ColumnDataSource, Div
from bokeh.layouts import column, row
import numpy as np
import matplotlib
from bokeh.plotting import figure

class LSBoundaryVisualizer:
    def __init__(self, shared_source, shared_resource, max_epoch, colors):
        self.source = shared_source
        self.shared_resource = shared_resource
        self.max_epoch = max_epoch
        self.colors = colors

        self.X = np.column_stack([self.source.data[feature] for feature in self.source.data if feature in ['x', 'y']])
        self.y = self.source.data['class']
        self.classes = np.unique(self.y)

        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1

        self.plot = figure(
            title="Evolving Boundary Visualization",
            width=600, height=600,
            x_range=(x_min, x_max),
            y_range=(y_min, y_max),
            tools="tap,box_select,box_zoom,reset,pan",
            active_drag="box_select"
        )

        initial_xs = shared_resource.data["xs"][0]
        initial_ys = shared_resource.data["ys"][0]
        self.boundary_source = ColumnDataSource(data={"xs": initial_xs, "ys": initial_ys})

        self.plot.scatter("x", "y", source=self.source, size="size", color="color", marker="marker")
        self.plot.multi_line(xs="xs", ys="ys", source=self.boundary_source, line_width=2, color="black")

        self.step_slider = Slider(start=0, end=self.max_epoch, value=0, step=1, title="Epoch")
        self.setup_callbacks()

    def setup_callbacks(self):
        self.step_slider.js_on_change("value", CustomJS(args={"source": self.source, 
                                                            "shared_resource": self.shared_resource,
                                                            "boundary_source": self.boundary_source,
                                                            }, 
        code="""
            var step = cb_obj.value;
            var shared_data = shared_resource.data;
            var step_index = shared_data["epoch"].indexOf(step);
            
            if (step_index !== -1) {
                source.data["size"] = shared_data["size"][step_index];
                boundary_source.data["xs"] = shared_data["xs"][step_index];
                boundary_source.data["ys"] = shared_data["ys"][step_index];

                source.change.emit();
                boundary_source.change.emit();
            }
        """))

    def get_layout(self):
        return column(self.plot, self.step_slider)