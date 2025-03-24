import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from torchvision import transforms
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Constants
IMAGE_SIZE = 256
class_names = ["Benign cases", "Malignant cases", "Normal cases"]
NUM_CLASSES = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the CNN model architecture
class ModifiedCNN(torch.nn.Module):
    def __init__(self):
        super(ModifiedCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = torch.nn.Linear(128 * (IMAGE_SIZE // 4) * (IMAGE_SIZE // 4), 512)
        self.fc2 = torch.nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 128 * (IMAGE_SIZE // 4) * (IMAGE_SIZE // 4))
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the trained model
model = ModifiedCNN().to(device)
model.load_state_dict(torch.load("lung_cancer_cnn1.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_ct_scan(image):
    """Check if the image is a CT scan based on grayscale intensity and color properties."""
    grayscale_image = image.convert("L")  
    np_image = np.array(grayscale_image)

    # Intensity statistics
    mean_intensity = np.mean(np_image)
    std_intensity = np.std(np_image)

    # Color analysis (CT scans should be grayscale, non-CTs have color variations)
    np_rgb = np.array(image)
    std_color = np.std(np_rgb, axis=(0, 1))  # Standard deviation per channel

    # Aspect ratio check (CT scans are often close to square)
    width, height = image.size
    aspect_ratio = width / height

    print(f"Mean Intensity: {mean_intensity}, Std Intensity: {std_intensity}, Std Color: {std_color}, Aspect Ratio: {aspect_ratio}")

    # **Decision Rules**
    if std_color.mean() > 80:  # Allow higher color variance
        print("⚠️ Warning: Image has high color variance but will still be processed.")
    if mean_intensity < 30 or mean_intensity > 200:  # Too bright or too dark
        return False
    if std_intensity < 10:  # Low contrast means it's probably not a CT scan
        return False
    if aspect_ratio < 0.8 or aspect_ratio > 1.2:  # CT scans are usually square or nearly square
        return False

    return True

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess the image
        image = Image.open(filepath).convert("L").convert("RGB")  # Force grayscale before processing

        # **Check if it's a valid CT scan**
        if not is_ct_scan(image):
            os.remove(filepath)
            return jsonify({"error": "Uploaded image is not a CT scan."}), 400

        transformed_image = transform(image).unsqueeze(0).to(device)

        # Perform prediction
        with torch.no_grad():
            outputs = model(transformed_image)
            predicted_label = torch.argmax(outputs, dim=1).item()

        # Delete the uploaded file after prediction
        os.remove(filepath)

        # Return prediction result to the front-end
        return jsonify({
            "prediction": class_names[predicted_label]
        })
    else:
        return jsonify({"error": "Invalid file type. Only PNG, JPG, and JPEG are allowed."}), 400

if __name__ == '__main__':
    app.run(debug=True)