from flask import Flask, render_template, request
from ultralytics import YOLO
import os
from PIL import Image
import torch

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load the YOLO model
model = YOLO('../runs/classify/train/weights/best.pt')   # Replace with your YOLO model path

# Check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if file and allowed_file(file.filename):
        # Save the uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Perform prediction
        results = model.predict(source=filepath)
        prediction = results[0].names[torch.argmax(results[0].probs.data).item()] 
        return render_template('index.html', prediction=prediction, image_url=filepath)

    return "Invalid file type", 400

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run()