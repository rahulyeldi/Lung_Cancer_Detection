import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import random
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Constants
IMAGE_SIZE = 256  
NUM_CLASSES = 3   
TEST_SAVE_DIR = "The IQ-OTHNCCD lung cancer dataset"
class_names = ["Benign cases", "Malignant cases", "Normal cases"]
label_mapping = {name: i for i, name in enumerate(class_names)}
idx_to_class = {v: k for k, v in label_mapping.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleCNN(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(128 * 64 * 64, 512)  
        self.fc2 = torch.nn.Linear(512, num_classes)  

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1) 
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)  
        return x

# Load the saved model
model = SimpleCNN(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load("lung_cancer_cnn1.pth",map_location=torch.device('cpu')))
model = model.to(device)

model.eval()  

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def classify_random_images_from_directory(test_dir, model, device):
    for class_name in class_names:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Directory not found for class: {class_name}")
            continue

        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            print(f"No images found for class: {class_name}")
            continue

        random_image = random.choice(image_files)
        image_path = os.path.join(class_dir, random_image)

        image = Image.open(image_path).convert("RGB")
        transformed_image = transform(image).unsqueeze(0).to(device)  

        model.eval()
        with torch.no_grad():
            outputs = model(transformed_image)
            predicted_label = torch.argmax(outputs, dim=1).item()

        plt.figure(figsize=(4, 4))
        plt.imshow(image)
        plt.title(f"True: {class_name}, Predicted: {idx_to_class[predicted_label]}")
        plt.axis("off")
        plt.show()

def evaluate_model(test_dir, model, device):
    y_true = []
    y_pred = []

    for class_name in class_names:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Directory not found for class: {class_name}")
            continue

        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for image_file in image_files:
            image_path = os.path.join(class_dir, image_file)

            # Load and preprocess the image
            image = Image.open(image_path).convert("RGB")
            transformed_image = transform(image).unsqueeze(0).to(device)

            # Predict using the model
            model.eval()
            with torch.no_grad():
                outputs = model(transformed_image)
                predicted_label = torch.argmax(outputs, dim=1).item()

            y_true.append(label_mapping[class_name]) 
            y_pred.append(predicted_label)           
    # Calculate metrics
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")


# model.load_state_dict(torch.load("lung_cancer_cnn.pth"))

classify_random_images_from_directory(TEST_SAVE_DIR, model, device)
evaluate_model(TEST_SAVE_DIR, model, device)
