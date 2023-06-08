import torch
import torch.nn.functional as F
from PIL import Image
import pytorch_lightning as pl
import torch.nn as nn
from torchvision import transforms as T
from torchvision import models
import matplotlib.pyplot as plt
import onnxruntime as ort
from glob import glob
import streamlit as st
import numpy as np
from torchmetrics.functional import accuracy
from torchmetrics import Accuracy

#Define the labels
labels = ['Defect', 'Non-Defect']

# Define the sample images
sample_images = {
    "Defect01": "pics/Defect/2.jpg",
    "Defect02": "pics/Defect/6.jpg",
    "Defect03": "pics/Defect/8.jpg",
    "Non-Defect01": "pics/nDefect/3.jpg",
    "Non-Defect02": "pics/nDefect/4.jpg",
    "Non-Defect03": "pics/nDefect/8.jpg"
}

class DefectResNet(pl.LightningModule):
    def __init__(self, n_classes=2):
        super(DefectResNet, self).__init__()
        
        # จำนวนของพันธุ์output (2)
        self.n_classes = n_classes

        #เปลี่ยน layer สุดท้าย
        self.backbone = models.resnet50(pretrained=True)
        # self.backbone = models.resnet152(pretrained=True)
        # self.backbone = models.vgg19(pretrained=True)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # เปลี่ยน fc layer เป็น output ขนาด 2
        self.backbone.fc = torch.nn.Linear(self.backbone.fc.in_features, n_classes) #For ResNet base mdoel
        # self.backbone.classifier[6] = torch.nn.Linear(self.backbone.classifier[6].in_features, n_classes) #For VGG bse model
        
        self.entropy_loss = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=2)

        self.save_hyperparameters(logger=False)

    def forward(self, x):
        preds = self.backbone(x)
        return preds

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.backbone(x)
        loss = self.entropy_loss(logits, y)
        y_pred = torch.argmax(logits, dim=1)
        self.log("train_loss", loss)
        self.log("train_acc", self.accuracy(y_pred, y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.backbone(x)
        loss = self.entropy_loss(logits, y)
        y_pred = torch.argmax(logits, dim=1)
        self.log("val_loss", loss)
        self.log("val_acc", self.accuracy(y_pred, y))
        return loss
        
    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return {
            "optimizer": self.optimizer,
            "monitor": "val_loss",
        }

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.backbone(x)
        loss = self.entropy_loss(logits, y)
        y_pred = torch.argmax(logits, dim=1)
        self.log("val_loss", loss)
        self.log("val_acc", self.accuracy(y_pred, y))
        return loss
    
    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        logits = self.backbone(x)
        loss = self.entropy_loss(logits, y)
        acc = accuracy(y_hat, y)
        return loss, acc

# Load the model on the appropriate device
loadmodel = DefectResNet()
def load_checkpoint(checkpoint):
    loadmodel.load_state_dict(checkpoint["state_dict"])
load_checkpoint(torch.load("models/model.ckpt"))
loadmodel.eval()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

def predict(image):
    image = transform(image).unsqueeze(0)

    # Perform the prediction
    with torch.no_grad():
        logits = loadmodel(image)
        probs = F.softmax(logits, dim=1)
        return probs

# Define the Streamlit app
def app():
    predictions = None
    st.title("Digital textile printing defect classification for industrial.")
    uploaded_file = st.file_uploader("Upload your image...", type=["jpg"])

    with st.expander("Or choose from sample here..."):
        sample = st.selectbox(label = "Select here", options = list(sample_images.keys()), label_visibility="hidden")   
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(sample_images["Defect01"], caption="Defect01", use_column_width=True)
        with col2:
            st.image(sample_images["Defect02"], caption="Defect02", use_column_width=True)
        with col3:
            st.image(sample_images["Defect03"], caption="Defect03", use_column_width=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(sample_images["Non-Defect01"], caption="Non-Defect01", use_column_width=True)
        with col2:
            st.image(sample_images["Non-Defect02"], caption="Non-Defect02", use_column_width=True)
        with col3:
            st.image(sample_images["Non-Defect03"], caption="Non-Defect03", use_column_width=True)

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
        # st.write(predictions)
        st.subheader(f'Predictions : {labels[torch.argmax(predictions[0]).item()]}')
        for pred, prob in zip(labels, predictions[0]):
            st.write(f"{pred}: {prob * 100:.2f}%")
            st.progress(prob.item())
    else:
        st.write("No predictions.")
    st.subheader("Credits")
    st.write("By : Settapun Laoaree | AI-Builders")
    st.markdown("Source : [Github](https://github.com/ShokulSet/DefectDetection-AIBuilders) [Hugging Face](https://huggingface.co/spaces/sh0kul/DefectDetection-Deploy)")

# Run the app
if __name__ == "__main__":
    app()