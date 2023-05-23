import torch
import torch.nn.functional as F
from PIL import Image
import pytorch_lightning as pl
import torch.nn as nn
from torchvision import transforms as T
from torchvision.models import resnet152
import matplotlib.pyplot as plt
import onnxruntime as ort
from glob import glob
import streamlit as st

# Load and transform the image
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

class ONNXModel(pl.LightningModule):
    def __init__(self, model_path, n_classes=2):
        super().__init__()
        self.onnx_session = ort.InferenceSession(model_path)
        self.input_name = self.onnx_session.get_inputs()[0].name
        
        # จำนวนของพันธุ์output (2)
        self.n_classes = n_classes

        #เปลี่ยน layer สุดท้าย
        self.backbone = resnet152(weights="ResNet152_Weights.DEFAULT")
        # self.backbone = models.resnet152(pretrained=True)
        # self.backbone = models.vgg19(pretrained=True)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # เปลี่ยน fc layer เป็น output ขนาด 2
        self.backbone.fc = torch.nn.Linear(self.backbone.fc.in_features, n_classes) #For ResNet base mdoel
        # self.backbone.classifier[6] = torch.nn.Linear(self.backbone.classifier[6].in_features, n_classes) #For VGG bse model
        self.onnx_session = ort.InferenceSession(model_path)
        self.input_name = self.onnx_session.get_inputs()[0].name

    def forward(self, x):
        preds = self.backbone(x)
        return preds

    #predict 1 รูป
    def predict_image(self, image):
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
        ])
        image = transform(image).unsqueeze(0)
        self.eval()
        with torch.no_grad():
            logits = self(image)
            probs = F.softmax(logits, dim=1)
            pred_label = torch.argmax(probs, dim=1)
        return pred_label.item()

# Load the model on the appropriate device
model = ONNXModel(model_path='/home/shokul/AIBuilder/deploy/DefectDetection-AIBuilders/models/model.onnx')
model = model.to("cpu")

def Predict(image_path):
    answer = glob(image_path)[0].split('/')[-2]
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    image = image.to("cpu")

    # Perform the prediction
    with torch.no_grad():
        logits = model(image)
        probs = F.softmax(logits, dim=1)
        pred_label = torch.argmax(probs, dim=1)

    #Testing is defect or not
    if pred_label.item() == 0:
        label = "nDefect"
    else:
        label = "Defect"
    
    #Show image and prediction
    plt.imshow(image.squeeze().permute(1, 2, 0))
    plt.title(f"Prediction: {label}")
    plt.suptitle(f"Answer: {answer}")
    plt.show()  

Predict('/home/shokul/AIBuilder/deploy/DefectDetection-AIBuilders/pics/nDefect/9.jpg')