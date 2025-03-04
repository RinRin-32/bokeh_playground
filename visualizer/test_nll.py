from bokeh.models import Button, CustomJS, Slider, ColumnDataSource, Div
from bokeh.layouts import column, row
import numpy as np
import matplotlib
from bokeh.plotting import figure

class TestNLLAnimation:
    def __init__(self, shared_source, shared_resource, max_epoch):
        self.source = shared_source
        self.shared_resource = shared_resource
        self.max_epoch = max_epoch

        # Extract initial values for epoch 0
        initial_epoch = 0
        initial_index = shared_resource.data["epoch"].index(initial_epoch) if initial_epoch in shared_resource.data["epoch"] else None

        if initial_index is not None:
            self.data_stream = ColumnDataSource(data={
                'epoch': [shared_resource.data["epoch"][initial_index]],
                'test_nll': [shared_resource.data["test_nll"][initial_index]],
                'estimated_nll': [shared_resource.data["estimated_nll"][initial_index]]
            })
        else:
            # If epoch 0 is not found, start with empty data
            self.data_stream = ColumnDataSource(data={'epoch': [], 'test_nll': [], 'estimated_nll': []})

        self.step_slider = Slider(start=0, end=self.max_epoch, value=0, step=1, title="Epoch")

        self.plot = self.create_plot()
        self.setup_callbacks()

        ## need to create the slider and temporary datasource here
        ## slider call backs need to update teh shared_source

    def setup_callbacks(self):
        self.step_slider.js_on_change("value", CustomJS(args={"source": self.data_stream,
                                                            "original": self.shared_resource,
                                                            "intermediate": self.source},
        code="""
            var step = cb_obj.value;
            var shared_data = original.data;
            var current_epochs = source.data["epoch"];
            
            if (current_epochs.length > 0) {
                var prev_max = Math.max(...current_epochs);
                
                if (step > prev_max) {
                    // Moving forward: Add all missing epochs
                    for (var s = prev_max + 1; s <= step; s++) {
                        var step_index = shared_data["epoch"].indexOf(s);
                        if (step_index !== -1) {
                            source.data["epoch"].push(shared_data["epoch"][step_index]);
                            source.data["test_nll"].push(shared_data["test_nll"][step_index]);
                            source.data["estimated_nll"].push(shared_data["estimated_nll"][step_index]);
                        }
                    }
                } else if (step < prev_max) {
                    // Moving backward: Pop until we match step
                    while (source.data["epoch"].length > 0 && Math.max(...source.data["epoch"]) > step) {
                        source.data["epoch"].pop();
                        source.data["test_nll"].pop();
                        source.data["estimated_nll"].pop();
                    }
                }
            } else {
                // First step, just add the current value
                var step_index = shared_data["epoch"].indexOf(step);
                if (step_index !== -1) {
                    source.data["epoch"].push(shared_data["epoch"][step_index]);
                    source.data["test_nll"].push(shared_data["test_nll"][step_index]);
                    source.data["estimated_nll"].push(shared_data["estimated_nll"][step_index]);
                }
            }
            intermediate.data["y"] = shared_data["y"][step]
            intermediate.data["x"] = shared_data["x"][step]

            intermediate.change.emit();
            
            source.change.emit();
        """))


    def create_plot(self):
        p = figure(title="Test vs Estimate NLL",
                   width=600, height=300,
                   x_range=(0, self.max_epoch),
                   )
        
        p.xaxis.axis_label = 'Epoch'
        p.yaxis.axis_label = 'NLL'

        p.line(x='epoch', y='test_nll', color='black', legend_label="Test NLL", source = self.data_stream)
        p.line(x='epoch', y='estimated_nll', color='red', legend_label="Estimated NLL", source = self.data_stream)

        p.y_range.only_visible = True

        return p
    
    def get_layout(self):
        return column(self.plot, self.step_slider)

