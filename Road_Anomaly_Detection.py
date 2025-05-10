from flask import Flask, render_template_string, request, url_for, redirect, session
import os
import uuid
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

# Load your model
model = load_model('model (2).h5', custom_objects={
    'mse': tf.keras.losses.MeanSquaredError(),
    'categorical_crossentropy': tf.keras.losses.CategoricalCrossentropy(),
    'accuracy': tf.keras.metrics.CategoricalAccuracy()
})

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Create static directories
os.makedirs('static/input', exist_ok=True)
os.makedirs('static/output', exist_ok=True)

# Class names (change as needed)
class_names = {
    1: 'Anomaly A',
    2: 'Anomaly B'
}

# HTML GUI
HTML_GUI = """
<!doctype html>
<html>
<head>
    <title>Road Anomaly Detection</title>
    <style>
        body {
            font-family: 'Segoe UI', sans-serif;
            background-color: #f5f7fa;
            margin: 0;
            padding: 40px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .container {
            background: white;
            padding: 30px 40px;
            border-radius: 12px;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.1);
            width: 100%;
            max-width: 800px;
            text-align: center;
        }

        h2 {
            margin-bottom: 20px;
            color: #333;
        }

        input[type="file"] {
            margin: 10px 0;
            padding: 10px;
            border: 2px dashed #aaa;
            background-color: #fff;
            border-radius: 8px;
            width: 100%;
            cursor: pointer;
        }

        button {
            padding: 10px 25px;
            border: none;
            background-color: #007bff;
            color: white;
            font-size: 16px;
            border-radius: 6px;
            cursor: pointer;
        }

        button:hover {
            background-color: #0056b3;
        }

        .preview-section {
            display: flex;
            justify-content: space-between;
            margin-top: 30px;
            gap: 20px;
            flex-wrap: wrap;
        }

        .image-box {
            flex: 1;
            min-width: 45%;
            background: #fafafa;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 10px;
        }

        .image-box h4 {
            margin-bottom: 10px;
            color: #555;
        }

        .image-box img {
            max-width: 100%;
            border-radius: 6px;
            border: 1px solid #ddd;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Road Anomaly Detection</h2>
        <form method="post" enctype="multipart/form-data">
            <h2>Upload Image</h2>
            <input type="file" name="input_image" accept="image/*" required><br>
            
            <h2>Upload txt file</h2>
            <input type="file" name="label_file" accept=".txt" required><br>
            
            <button type="submit">Submit</button>
        </form>


        {% if input_image and result_image %}
        <div class="preview-section">
            <div class="image-box">
                <h4>Before</h4>
                <img src="{{ url_for('static', filename='input/' + input_image) }}" alt="Input Image">
            </div>
            <div class="image-box">
                <h4>After</h4>
                <img src="{{ url_for('static', filename='output/' + result_image) }}" alt="Output Image">
            </div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

# Parse YOLO label file
def parse_label_file(filepath):
    labels = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls, x, y, w, h = map(float, parts)
                labels.append([cls, x, y, w, h])
    return labels

# Draw boxes on image
def draw_boxes(image, boxes, color, label_prefix=""):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    width, height = image.size

    for box in boxes:
        cls, cx, cy, w, h = box
        x1 = int((cx - w / 2) * width)
        y1 = int((cy - h / 2) * height)
        x2 = int((cx + w / 2) * width)
        y2 = int((cy + h / 2) * height)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{label_prefix}{class_names.get(int(cls), str(int(cls)))}"
        draw.text((x1, y1 - 10), label, fill=color, font=font)

# Main image processing 
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def process_image(input_path, output_path, original_labels):
    original_img = Image.open(input_path).convert('RGB')
    model_input = original_img.resize((224, 224))

    img_array = np.array(model_input) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    print("Prediction:", prediction, "\n" , "Original" , original_labels )

    # Use only the second prediction array assuming it's (cx, cy, w, h)
    if isinstance(prediction, list) and len(prediction) >= 2:
        bbox_array = prediction[1]  # assuming second is bbox
    else:
        bbox_array = prediction  # fallback

    # Convert to list of boxes: [class_id, cx, cy, w, h]
    predicted_boxes = []
    for bbox in bbox_array:
        cx, cy, w, h = bbox
        predicted_boxes.append([1, float(cx), float(cy), float(w), float(h)])

    # Draw original and predicted boxes
    before_image = original_img.copy()
    after_image = original_img.copy()

    draw_boxes(before_image, original_labels, color='green', label_prefix="GT: ")
    draw_boxes(after_image, predicted_boxes, color='red', label_prefix="Pred: ")

    before_image.save(input_path)
    after_image.save(output_path)


# Home route
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'input_image' in request.files and 'label_file' in request.files:
            image_file = request.files['input_image']
            label_file = request.files['label_file']

            if image_file.filename != '' and label_file.filename != '':
                unique_id = str(uuid.uuid4())[:8]
                image_filename = secure_filename(image_file.filename)
                label_filename = f"{unique_id}_labels.txt"

                input_image_filename = f"{unique_id}_{image_filename}"
                input_path = os.path.join('static/input', input_image_filename)
                label_path = os.path.join('static/input', label_filename)

                image_file.save(input_path)
                label_file.save(label_path)

                original_labels = parse_label_file(label_path)
                result_image_filename = f"result_{input_image_filename}"
                output_path = os.path.join('static/output', result_image_filename)

                process_image(input_path, output_path, original_labels)

                session['input_image'] = input_image_filename
                session['result_image'] = result_image_filename

                return redirect(url_for('home'))

    input_image = session.pop('input_image', None)
    result_image = session.pop('result_image', None)

    return render_template_string(HTML_GUI, input_image=input_image, result_image=result_image)

if __name__ == '__main__':
    app.run(debug=True)
