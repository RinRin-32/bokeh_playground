from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models import HoverTool, ColumnDataSource, Div, CustomJS, Select
from visualizer.evolvingmpe import EvolvingMemoryMapVisualizer
from collections import defaultdict

class LabelNoisePlot:
    def __init__(self, shared_source, plot_name, show_mm=False):
        self.shared_source = shared_source
        self.show_mm = show_mm
        self.plot_name = plot_name
        
        self.unique_labels = list(set(shared_source.data['label']))
        self.unique_labels.sort()
        self.unique_labels.insert(0, "All")  # Add an "All" option
        
        self.filtered_source = ColumnDataSource(data=self.shared_source.data.copy())
        self.plot = self.create_plot()
        self.selected_source = ColumnDataSource(data=dict(img=[], label=[]))
        
        self.image_display = Div(
            text="<h3>Selected Images:</h3>", 
            width=500, height=600, 
            stylesheets=[""" .scroll-box { overflow-y: auto; max-height: 600px; padding: 10px; } """], 
            css_classes=["scroll-box"]
        )
        
        self.dropdown = Select(title="Select Class:", value="All", options=self.unique_labels)
        
        self.callback = CustomJS(args=dict(source=self.filtered_source, selected_source=self.selected_source, display=self.image_display), code="""
            var selected_indices = source.selected.indices;
            var imgs = [];
            var labels = [];
            
            for (var i = 0; i < source.data['x'].length; i++) {
                source.data['color'][i] = 'grey';  // Reset all to grey
            }
            
            for (var i = 0; i < selected_indices.length; i++) {
                source.data['color'][selected_indices[i]] = 'red';  // Highlight selected points
                imgs.push(source.data['img'][selected_indices[i]]);
                labels.push(source.data['label'][selected_indices[i]]);
            }
            
            source.change.emit();
            
            // Group images by class
            var grouped = {};
            for (var i = 0; i < imgs.length; i++) {
                var label = labels[i];
                if (!(label in grouped)) {
                    grouped[label] = [];
                }
                grouped[label].push(imgs[i]);
            }
            
            // Generate HTML with wrapping rows (max 10 images per row)
            var html = "<div style='display: flex; flex-direction: column; gap: 10px;'>";
            var sorted_labels = Object.keys(grouped).sort((a, b) => a - b); // Sort classes numerically

            for (var j = 0; j < sorted_labels.length; j++) {
                var label = sorted_labels[j];
                var images = grouped[label];

                html += `<div><b>Class ${label}</b></div>`;
                html += "<div style='display: flex; flex-direction: column; gap: 5px;'>";

                // Break images into lines of max 10 per row
                for (var k = 0; k < images.length; k += 10) {
                    html += "<div style='display: flex; flex-wrap: wrap; gap: 5px;'>";
                    for (var m = k; m < Math.min(k + 10, images.length); m++) {
                        html += `<img src='data:image/png;base64,${images[m]}' width='56' height='56'>`;
                    }
                    html += "</div>";  // Close row
                }

                html += "</div>";  // Close class container
            }

            html += "</div>";
            display.text = '<div class="scroll-box">' + html + '</div>';
        """)

        
        self.filtered_source.selected.js_on_change("indices", self.callback)
        self.mm_setup()

    def mm_setup(self):
        if self.show_mm:
            self.dropdown.js_on_change("value", CustomJS(args=dict(source=self.shared_source, filtered_source=self.filtered_source, dropdown=self.dropdown), code="""
                var selected_class = dropdown.value;
                var new_data = {x: [], y: [], label: [], img: [], color: [], bls: [], bpe:[], size:[], marker:[], alpha:[]};
                
                for (var i = 0; i < source.data['label'].length; i++) {
                    if (selected_class === 'All' || source.data['label'][i] === selected_class) {
                        new_data['x'].push(source.data['x'][i]);
                        new_data['y'].push(source.data['y'][i]);
                        new_data['label'].push(source.data['label'][i]);
                        new_data['img'].push(source.data['img'][i]);
                        new_data['color'].push(source.data['color'][i]);
                        new_data['bls'].push(source.data['bls'][i]);
                        new_data['bpe'].push(source.data['bpe'][i]);
                        new_data['size'].push(source.data['size'][i]);
                        new_data['marker'].push(source.data['marker'][i]);
                        new_data['alpha'].push(source.data['alpha'][i]);                      
                    }
                }
                
                filtered_source.data = new_data;
                filtered_source.change.emit();
            """))
            self.memory_map_visualizer = EvolvingMemoryMapVisualizer(self.filtered_source)
            self.memory_map_layout = column(self.memory_map_visualizer.get_layout(), width=500)
        else:
            self.dropdown.js_on_change("value", CustomJS(args=dict(source=self.shared_source, filtered_source=self.filtered_source, dropdown=self.dropdown), code="""
                var selected_class = dropdown.value;
                var new_data = {x: [], y: [], label: [], img: [], color: []};
                
                for (var i = 0; i < source.data['label'].length; i++) {
                    if (selected_class === 'All' || source.data['label'][i] === selected_class) {
                        new_data['x'].push(source.data['x'][i]);
                        new_data['y'].push(source.data['y'][i]);
                        new_data['label'].push(source.data['label'][i]);
                        new_data['img'].push(source.data['img'][i]);
                        new_data['color'].push(source.data['color'][i]);                   
                    }
                }
                
                filtered_source.data = new_data;
                filtered_source.change.emit();
            """))

    def create_plot(self):
        p = figure(width=800, height=600, tools="reset,save,box_select",
           title=f"{self.plot_name} Label Noise Distribution",
           x_axis_label="Examples", y_axis_label=r"Label Noise ||ε||₂")
        p.title.text_font_size = "25px"
        p.title.align = 'center'
        p.xaxis.visible = False
        p.yaxis.major_label_text_font_size = "15pt"
        p.yaxis.axis_label_text_font_size = "20pt"
        p.xaxis.axis_label_text_font_size = "20pt"

        if self.show_mm:
            p.scatter("x", "y", source=self.filtered_source, size=6, color="color", legend_label="Data", fill_alpha="alpha", marker="marker")
        else:
            p.scatter("x", "y", source=self.filtered_source, size=6, color="color", legend_label="Data", fill_alpha=0.6)

        hover = HoverTool(tooltips="""
            <div>
                <img src="data:image/png;base64,@img" width="28" height="28"></img>
                <br>
                <b>Label:</b> @label
            </div>
        """)
        p.add_tools(hover)

        return p
    
    def generate_html(self, img_list, label_list):
        items = [f'<div><img src="data:image/png;base64,{img}" width="56" height="56"><br>Label: {label}</div>'
                for img, label in zip(img_list, label_list)]
        return "<div style='display: flex; flex-wrap: wrap; gap: 10px;'>" + "".join(items) + "</div>"
    
    def get_layout(self):
        if self.show_mm:
            return column(row(self.memory_map_layout, self.plot, self.image_display), self.dropdown)
        else:
            return column(row(self.plot, self.image_display), self.dropdown)