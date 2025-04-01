from bokeh.models import ColumnDataSource, CustomJS, Div, Slider, Spacer
from bokeh.layouts import column, row

class Sample:
    def __init__(self, shared_source, shared_resource, plot_name, y_range, n_sample, max_epoch, default_color='white', display_mode='column'):
        self.shared_source = shared_source
        self.shared_resource = shared_resource
        self.plot_name = plot_name
        self.n_sample = n_sample
        self.max_epoch = max_epoch
        self.default_color = default_color
        self.display_mode = display_mode

        self.y_min = y_range[0]
        self.y_max = y_range[1]

        # Slider to control the epoch
        self.step_slider = Slider(start=0, end=self.max_epoch, value=0, step=1, title="Epoch")

        # Divs to display the images
        self.image_displays = self.create_image_displays()

        # Generate images initially on creation
        self.update_images()

        # Set up callbacks for slider updates
        self.setup_callbacks()

    def create_image_displays(self):
        if self.display_mode == row:
            width, height = 300, 300
        else:
            width, height = 400, 200
        return {
            "high_noise_images": Div(text="<h3>High Label Noise Examples:</h3>", width=width, height=height, css_classes=["scroll-box"]),
            "low_noise_images": Div(text="<h3>Low Label Noise Examples:</h3>", width=width, height=height, css_classes=["scroll-box"])
        }

    def update_images(self):
        # Initial image plotting (same logic as callback)
        shared_data = self.shared_resource.data
        images = self.shared_source.data["img"]
        labels = self.shared_source.data["label"]
        x_values = self.shared_source.data["x"]
        y_values = self.shared_source.data["y"]
        indices = len(x_values)

        high_noise_images_by_label = {}
        low_noise_images_by_label = {}

        # Iterate through data points and sort by noise levels
        for i in range(indices):
            label = labels[i]
            imgTag = "<img src='data:image/png;base64," + images[i] + "' width='32' height='32'>"
            
            if y_values[i] > 0.1:  # High noise
                if label not in high_noise_images_by_label:
                    high_noise_images_by_label[label] = []
                if len(high_noise_images_by_label[label]) < 5:
                    high_noise_images_by_label[label].append(imgTag)
            elif y_values[i] > 0:  # Low noise
                if label not in low_noise_images_by_label:
                    low_noise_images_by_label[label] = []
                if len(low_noise_images_by_label[label]) < 5:
                    low_noise_images_by_label[label].append(imgTag)

        # Generate the HTML for both high and low noise categories
        def generate_html(images_by_label, title):
            if self.display_mode:
                html = "<h3>" + title + "</h3><div style='display: flex; flex-direction: row; align-items: flex-start;'>"
            else:
                html = "<h3>" + title + "</h3>"
            sortedLabels = sorted(images_by_label.keys())
            for label in sortedLabels:
                if self.display_mode == 'column':  
                    html += f"<div style='display: flex; flex-direction: column; align-items: center; margin-right: 1px;'>"
                    html += f"".join(images_by_label[label]) + "</div>"
                else:
                    html += f"<div style='display: flex; flex-wrap: wrap; gap: 0px; margin-bottom: 1px;'" + "".join(images_by_label[label]) + "</div>"
            html += "</div>"
            return html

        self.image_displays["high_noise_images"].text = '<div class="scroll-box">' + generate_html(high_noise_images_by_label, "High Label Noise Examples") + '</div>'
        self.image_displays["low_noise_images"].text = '<div class="scroll-box">' + generate_html(low_noise_images_by_label, "Low Label Noise Examples") + '</div>'

    def setup_callbacks(self):
        self.step_slider.js_on_change("value", CustomJS(args={"original": self.shared_resource, "source": self.shared_source, "image_displays": self.image_displays, "display_mode": self.display_mode},
        code="""
            var step = cb_obj.value;
            var shared_data = original.data;

            source.data["y"] = shared_data["y"][step];
            source.data["x"] = shared_data["x"][step];
            source.data["img"] = shared_data["img"][step];
            source.data["label"] = shared_data["label"][step];

            var images = source.data["img"];
            var labels = source.data["label"];
            var y_values = source.data["y"];
            var indices = images.length;

            var high_noise_images_by_label = {};
            var low_noise_images_by_label = {};

            for (var i = 0; i < indices; i++) {
                var label = labels[i];
                var imgTag = "<img src='data:image/png;base64," + images[i] + "' width='32' height='32'>";
                
                if (y_values[i] > 0.1) {  
                    if (!(label in high_noise_images_by_label)) {
                        high_noise_images_by_label[label] = [];
                    }
                    if (high_noise_images_by_label[label].length < 5){
                        high_noise_images_by_label[label].push(imgTag);
                    }
                } else if (y_values[i] > 0) {  
                    if (!(label in low_noise_images_by_label)) {
                        low_noise_images_by_label[label] = [];
                    }
                    if (low_noise_images_by_label[label].length < 5){
                        low_noise_images_by_label[label].push(imgTag);
                    }
                }
            }

            function generate_html(images_by_label, title) {
                if (display_mode === 'row') {
                    var html = "<h3>" + title + "</h3><div style='display: flex; flex-direction: row; gap: 20px;'>";
                    var sortedLabels = Object.keys(images_by_label).sort();
                    sortedLabels.forEach(function(label) {
                        html += "<div style='display: flex; flex-wrap: wrap; gap: 0px; margin-bottom: 1px;'><b>" + label + "</b> " + images_by_label[label].join("") + "</div>";
                    });
                    html += "</div>";  // Close the row container
                    return html;
                }else{
                    var html = "<h3>" + title + "</h3><div style='display: flex; flex-direction: row; align-items: flex-start;'>";
                    var sortedLabels = Object.keys(images_by_label).sort();
        
                    sortedLabels.forEach(function(label) {
                        html += "<div style='display: flex; flex-direction: column; align-items: center; margin-right: 1px;'>";
                        html += images_by_label[label].join("") + "</div>";
                    });

                    html += "</div>";  // Close row container
                    return html;
                }
            }

            image_displays["high_noise_images"].text = '<div class="scroll-box">' + generate_html(high_noise_images_by_label, "High Label Noise Examples") + '</div>';
            image_displays["low_noise_images"].text = '<div class="scroll-box">' + generate_html(low_noise_images_by_label, "Low Label Noise Examples") + '</div>';
        """))

    def get_layout(self):
        # Add some space between the image divs and the slider
        if self.display_mode == row:
            return column(row(self.image_displays["high_noise_images"], self.image_displays["low_noise_images"]), Spacer(height=70), self.step_slider)
        else:
            return column(self.image_displays["high_noise_images"], self.image_displays["low_noise_images"], Spacer(height=70), self.step_slider)