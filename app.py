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
import numpy as np

#Define the labels
labels = ['Defect', 'nDefect']

# Define the sample images
sample_images = {
    "Defect01": "pics/Defect/0.jpg",
    "Defect02": "pics/Defect/1.jpg",
    "Defect03": "pics/Defect/2.jpg",
    "Non-Defect01": "pics/nDefect/0.jpg",
    "Non-Defect02": "pics/nDefect/1.jpg",
    "Non-Defect03": "pics/nDefect/2.jpg"
}

class ONNXModel(pl.LightningModule):
    def __init__(self, model_path, n_classes=2):
        super().__init__()
        self.onnx_session = ort.InferenceSession(model_path)
        self.input_name = self.onnx_session.get_inputs()[0].name
        self.n_classes = n_classes
        #Change the last layer
        self.backbone = resnet152(weights="ResNet152_Weights.DEFAULT")
        # self.backbone = models.resnet152(pretrained=True)
        # self.backbone = models.vgg19(pretrained=True)
        for param in self.backbone.parameters():
            param.requires_grad = False
        #Change the last layer to 2 classes
        self.backbone.fc = torch.nn.Linear(self.backbone.fc.in_features, n_classes) #For ResNet base mdoel
        # self.backbone.classifier[6] = torch.nn.Linear(self.backbone.classifier[6].in_features, n_classes) #For VGG bse model
        self.onnx_session = ort.InferenceSession(model_path)
        self.input_name = self.onnx_session.get_inputs()[0].name
    def forward(self, x):
        preds = self.backbone(x)
        return preds

# Load the model on the appropriate device
model = ONNXModel(model_path='models/model.onnx')
model = model.to("cpu")

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

def predict(image):
    image = transform(image).unsqueeze(0)
    image = image.to("cpu")

    # Perform the prediction
    with torch.no_grad():
        logits = model(image)
        probs = F.softmax(logits, dim=1)
        return probs

# Define the Streamlit app
def app():
    predictions = None
    st.title("Defect Classification")

    uploaded_file = st.file_uploader("Upload your image...", type=["jpg"])

    with st.expander("Or choose from sample here..."):
        
        st.header("Sample Defect Images")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(sample_images["Defect01"], caption="Defect01", use_column_width=True)
        with col2:
            st.image(sample_images["Defect02"], caption="Defect02", use_column_width=True)
        with col3:
            st.image(sample_images["Defect03"], caption="Defect03", use_column_width=True)
        st.header("Sample Non-Defect Images")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(sample_images["Non-Defect01"], caption="Non-Defect01", use_column_width=True)
        with col2:
            st.image(sample_images["Non-Defect02"], caption="Non-Defect02", use_column_width=True)
        with col3:
            st.image(sample_images["Non-Defect03"], caption="Non-Defect03", use_column_width=True)

        sample = st.selectbox(label = "Select here", options = list(sample_images.keys()), label_visibility="hidden")   

    # If an image is uploaded, make a prediction on it
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        predictions = predict(image)
    elif sample:    
        image = Image.open(sample_images[sample])
        st.image(image, caption=sample.capitalize() + " Image", use_column_width=True)
        predictions = predict(image)
      
    # Show  predictions with their probabilities
    if predictions is not  None:
        st.write(predictions)
        for pred, prob in zip(labels, predictions[0]):
            st.write(f"{pred}: {prob * 100:.2f}%")
            st.progress(prob.item())
    else:
        st.write("No predictions.")


# Run the app
if __name__ == "__main__":
    app()