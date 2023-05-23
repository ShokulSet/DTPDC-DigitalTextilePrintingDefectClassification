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

labels = ['Defect', 'nDefect']

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
model = ONNXModel(model_path='/home/shokul/AIBuilder/deploy/DefectDetection-AIBuilders/models/model.onnx')
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
      tab1, tab2 = st.tabs(["Defect", "Non-Defect"])
      with tab1:
        st.header("Defect")
        st.image("https://static.streamlit.io/examples/cat.jpg")
      with tab2:
        st.header("Non-Defect")
        st.image("https://static.streamlit.io/examples/dog.jpg")

    # If an image is uploaded, make a prediction on it
    if uploaded_file is not None:
      image = Image.open(uploaded_file)
      st.image(image, caption="Uploaded Image.", use_column_width=True)
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