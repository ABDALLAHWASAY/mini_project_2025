import os
import json
import uuid
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

MODEL_PATH = '_model/skin_model.h5'
LABELS_PATH = '_model/labels.json'
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load model and label dictionary
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = load_model(MODEL_PATH)

with open(LABELS_PATH, 'r') as f:
    class_indices = json.load(f)
labels = {v: k for k, v in class_indices.items()}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', error="No image uploaded")
    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', error="No image selected")
    if not allowed_file(file.filename):
        return render_template('index.html', error="Invalid file type (only PNG, JPG, JPEG allowed)")

    # Generating  filename
    ext = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(filepath)

    try:
        # Loading image and preprocessing
        img = load_img(filepath, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)[0]
        pred_class = labels[np.argmax(preds)]
        confidence = float(np.max(preds))
    except Exception as e:
        return render_template('index.html', error=f"Prediction failed: {str(e)}")

    return render_template('result.html', image_path=filepath, prediction=pred_class, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=False)
