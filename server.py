# -*- coding: utf-8 -*-

import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, render_template, request
import base64
import torchvision.models as models

app = Flask(__name__)

class_names = ['광안리해수욕장', '남포동 BIFF 거리', '범어사(부산)',
               '벡스코(BEXCO)', '부산 감천문화마을', '부산 송도해상케이블카',
               '부산 암남공원', '부산 자갈치시장', '부산광안대교',
               '부산다이아몬드타워', '송도 구름산책로(스카이워크)', '오륙도 (부산 국가지질공원)',
               '용두산공원', '이기대 (부산 국가지질공원)', '재한유엔기념공원 (UN기념공원)',
               '태종대 전망대', '해동 용궁사(부산)', '해운대 달맞이길',
               '해운대 동백섬', '해운대해수욕장']

model = models.resnet34(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))

model.load_state_dict(torch.load(
    'model/image_classification.pth', map_location=torch.device('cpu')))
model.eval()


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(image)


@app.route("/", methods=['GET', 'POST'])
@app.route("/upload", methods=['POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        image_bytes = file.read()

        preprocessed_image = preprocess_image(image_bytes=image_bytes)

        with torch.no_grad():
            input_tensor = preprocessed_image.unsqueeze(0)
            outputs = model(input_tensor)
            _, preds = torch.max(outputs, 1)
            class_name = class_names[preds[0].item()]

        result = {
            'class': class_name,
            'image_base64': base64.b64encode(image_bytes).decode('utf-8')
        }

        return render_template('index.html', result=result)
    else:
        return render_template('index.html', result=None)


if __name__ == '__main__':
    app.run()
