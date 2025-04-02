from bokeh.models import ColumnDataSource, CustomJS, Div, Slider, Spacer, Button
from bokeh.layouts import column, row

class Sample:
    def __init__(self, shared_source, shared_resource, plot_name, y_range, n_sample, max_epoch, default_color='white', display_mode='row'):
        self.shared_source = shared_source
        self.shared_resource = shared_resource
        self.plot_name = plot_name
        self.n_sample = n_sample
        self.max_epoch = max_epoch
        self.default_color = default_color
        self.display_mode = display_mode  # "column" or "row"

        self.play_pause_button = Button(label="Play")

        self.y_min = y_range[0]
        self.y_max = y_range[1]

        # Slider to control the epoch
        self.step_slider = Slider(start=0, end=self.max_epoch, value=0, step=1, title="Epoch")

        # Single Div to display the images ordered by label and noise
        self.image_display = Div(text="<h3>Label Noise Examples:</h3>", css_classes=["scroll-box"])

        # Generate images initially on creation
        self.update_images()

        # Set up callbacks for slider updates
        self.setup_callbacks()

    def update_images(self):
        # Initial image plotting (same logic as callback)
        # Use the shared_source data
        images = self.shared_source.data["img"]
        labels = self.shared_source.data["label"]
        y_values = self.shared_source.data["y"]
        indices = len(images)

        # Group images by label with their noise value
        images_by_label = {}
        for i in range(indices):
            label = labels[i]
            noise = y_values[i]
            imgTag = "<img src='data:image/png;base64," + images[i] + "' width='10' height='10'>"
            if label not in images_by_label:
                images_by_label[label] = []
            images_by_label[label].append((noise, imgTag))
        
        # For each label, sort images by noise descending (high to low)
        for label in images_by_label:
            images_by_label[label].sort(key=lambda tup: tup[0], reverse=True)

        # Generate the HTML for each label
        def generate_html(images_by_label, title):
            html = "<h3>" + title + "</h3><div style='display: flex; flex-direction: column;'>"
            # Sort labels in ascending order (or adjust as needed)
            for label in sorted(images_by_label.keys()):
                html += f"<div style='margin-bottom: 1px;'"
                # Append each image (already sorted by noise descending)
                for (_, imgTag) in images_by_label[label]:
                    html += imgTag
                html += "</div>"
            html += "</div>"
            return html

        self.image_display.text = generate_html(images_by_label, "Label Noise Examples")

    def setup_callbacks(self):
        self.step_slider.js_on_change("value", CustomJS(args={
            "original": self.shared_resource, 
            "source": self.shared_source, 
            "image_display": self.image_display,
            "display_mode": self.display_mode
        },
        code="""
            // Update the shared_source data based on the slider value (epoch)
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

            // Group images by label along with their noise value
            var images_by_label = {};
            for (var i = 0; i < indices; i++) {
                var label = labels[i];
                var noise = y_values[i];
                var imgTag = "<img src='data:image/png;base64," + images[i] + "' width='10' height='10'>";
                if (!(label in images_by_label)) {
                    images_by_label[label] = [];
                }
                images_by_label[label].push({noise: noise, tag: imgTag});
            }
            
            // For each label, sort by noise descending
            for (var label in images_by_label) {
                images_by_label[label].sort(function(a, b) {
                    return b.noise - a.noise;
                });
            }
            
            // Generate HTML content
            function generate_html(images_by_label, title) {
                var html = "<h3>" + title + "</h3><div style='display: flex; flex-direction: column;'>";
                var sortedLabels = Object.keys(images_by_label).sort();
                sortedLabels.forEach(function(label) {
                    html += "<div style='margin-bottom: 1px;'";
                    images_by_label[label].forEach(function(obj) {
                        html += obj.tag;
                    });
                    html += "</div>";
                });
                html += "</div>";
                return html;
            }

            image_display.text = generate_html(images_by_label, "Label Noise Examples");
            source.change.emit();
        """))

        self.play_pause_button.js_on_click(CustomJS(args={"slider": self.step_slider, "button": self.play_pause_button, "max_epoch": self.max_epoch}, code="""
            var step = slider.value;
            var is_playing = button.label == "Pause";
            var is_at_end = step >= max_epoch;
            
            if (is_at_end) {
                button.label = "Restart";
                slider.value = 0;
                step = 0;
            }
            
            if (is_playing) {
                button.label = "Play";
                clearTimeout(slider._timeout);
            } else {
                button.label = "Pause";
                function animate() {
                    if (step < max_epoch) {
                        step += 1;
                        slider.value = step;
                        slider._timeout = setTimeout(animate, 500);
                    } else {
                        button.label = "Restart";
                    }
                }
                animate();
            }
        """))

    def get_layout(self):
        # Layout: single Div for images and slider below
        if self.display_mode == "row":
            return column(row(self.image_display), self.step_slider,  self.play_pause_button)
        else:
            return column(self.image_display, self.step_slider, self.play_pause_button)