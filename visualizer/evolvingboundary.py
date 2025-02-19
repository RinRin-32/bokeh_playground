from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Button, Slider, Div, CustomJS
from bokeh.io import curdoc
import numpy as np

class EvolvingBoundaryVisualizer:
    def __init__(self, shared_source, shared_resource, mod1, steps, colors, batches=4, max_steps=30):
        self.source = shared_source
        self.shared_resource = shared_resource
        self.batches = batches
        self.mod1 = mod1
        self.steps = steps
        self.colors = colors

        self.X = np.column_stack([self.source.data[feature] for feature in self.source.data if feature in ['x', 'y']])
        self.y = self.source.data['class']

        self.classes = np.unique(self.y)

        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1

        self.step = 0
        self.max_steps = max_steps
        self.message_div = Div(text="", width=400, height=50)

        self.plot = figure(
            title="Evolving Boundary Visualization",
            width=600, height=600,
            x_range=(x_min, x_max),
            y_range=(y_min, y_max),
            tools="tap,box_select,box_zoom,reset,pan",
            active_drag="box_select"
        )
        
        self.boundary_source = ColumnDataSource(data={"xs": [], "ys": []
                                                      #, "prev_xs": [], "prev_ys": []
                                                      })

        self.plot.scatter("x", "y", source=self.source, size="size", color="color", marker="marker", alpha="alpha")
        self.plot.multi_line(xs="xs", ys="ys", source=self.boundary_source, line_width=2, color="black")
        #self.plot.multi_line(xs="prev_xs", ys="prev_ys", source=self.source, line_width=2, color="grey")

        
        self.step_slider = Slider(start=0, end=self.max_steps, value=0, step=1, title="Epoch Step")
        self.play_pause_button = Button(label="Play")
        self.reset_button = Button(label="Reset", button_type="danger")
        
        self.setup_callbacks()

    def setup_callbacks(self):
        self.step_slider.js_on_change("value", CustomJS(args={"source": self.source, 
                                                            "shared_resource": self.shared_resource,
                                                            "boundary_source": self.boundary_source}, 
        code="""
            var step = cb_obj.value;
            var shared_data = shared_resource.data;
            var step_index = shared_data["step"].indexOf(step);
            
            if (step_index !== -1) {
                source.data["bls"] = shared_data["bls"][step_index];
                source.data["bpe"] = shared_data["bpe"][step_index];
                source.data["sensitivities"] = shared_data["sensitivities"][step_index];
                source.data["softmax_deviations"] = shared_data["softmax_deviations"][step_index];

                boundary_source.data["xs"] = shared_data["xs"][step_index];
                boundary_source.data["ys"] = shared_data["ys"][step_index];
                source.change.emit();
                boundary_source.change.emit();
            }
        """))

        self.play_pause_button.js_on_click(CustomJS(args={"slider": self.step_slider}, code="""
            var step = slider.value;
            function animate() {
                if (step < slider.end) {
                    step += 1;
                    slider.value = step;
                    setTimeout(animate, 200);
                }
            }
            animate();
        """))

        self.reset_button.js_on_click(CustomJS(args={"slider": self.step_slider}, code="""
            slider.value = 0;
        """))
    
    def get_layout(self):
        return column(
            self.plot,
            self.message_div,
            row(self.play_pause_button, self.reset_button),
            self.step_slider
        )