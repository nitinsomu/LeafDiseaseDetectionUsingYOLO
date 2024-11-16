from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import torch

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

model = YOLO('../runs/classify/train/weights/best.pt') 

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

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        results = model(filepath)
        prediction = results[0].names[torch.argmax(results[0].probs.data).item()] 
        return render_template('index.html', prediction=prediction, image_url=filepath)

    return "Invalid file type", 400

if __name__ == '__main__':
    app.run(debug=True)
