import os
import torch
from flask import Flask, render_template, request, jsonify
from PIL import Image
from torchvision import models, transforms
from collections import OrderedDict

app = Flask(__name__)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

model_path = r'C:\Users\User\Downloads\Sortify Temporary\Sortify\model.pth'

resnet_model = models.resnet50(pretrained=False)
resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, len(class_names))

state_dict = torch.load(model_path, map_location=torch.device('cpu'))
new_state_dict = OrderedDict((key.replace('network.', ''), value) for key, value in state_dict.items())
resnet_model.load_state_dict(new_state_dict)
resnet_model.eval()

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    try:
        imagefile = request.files['imagefile']
        IMAGE_DIR = r'C:\Users\User\Downloads\Sortify Temporary\Sortify\IMAGES'
        os.makedirs(IMAGE_DIR, exist_ok=True)

        image_path = os.path.join(IMAGE_DIR, imagefile.filename)
        imagefile.save(image_path)

        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = resnet_model(image)
            class_scores = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_class = torch.max(class_scores, dim=0)

        prediction = class_names[predicted_class.item()]
        accuracy = round(confidence.item() * 100, 2)

        return render_template('index.html', prediction=prediction, Accuracy=accuracy)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(port=3000, debug=True)