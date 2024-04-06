import streamlit as st
from PIL import Image

import torch
from cnn import SimpleCNN,CNN
from torchvision import transforms
import torchvision
from PIL import Image
import numpy as np

def load_model():
   dict = torch.load("model_48.pth",map_location=torch.device('cpu'))
   # Cargar el modelo
   num_classes = 15
   list_dropouts =  [0.4154439238682205,0.21085094262884393,0.4313465383566946, 0.32912701708030556,0.4898772544775635]
   list_neuronas = [199,194,160,114,171]
   modelo = SimpleCNN(torchvision.models.resnet50(), n_classes=num_classes,n_layers=5, unfreezed_layers=7,list_dropouts=list_dropouts,list_neuronas_salida=list_neuronas)

   modelo.load_state_dict(dict)
   modelo.eval()
   return modelo

model = load_model()

# Función para predecir la imagen
def predict(image,model):
    
    img_size = 224
    # Define image transformations
    transformaciones = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        transforms.Resize(size =(img_size,img_size)),
    ])


    input_tensor = transformaciones(image)# Add batch dimension
    input_tensor = input_tensor.unsqueeze(0)

    class_to_index = {0: 'Inside city', 1: 'Kitchen', 2: 'Office', 3: 'Store', 4: 'Street', 5: 'Suburb', 6: 'Highway', 7: 'Coast', 8: 'Mountain', 9: 'Open country', 10: 'Industrial', 11: 'Forest', 12: 'Tall building', 13: 'Bedroom', 14: 'Living room'}
    outputs = model(input_tensor)
    salida = class_to_index[outputs.argmax(1).item()]
    return salida

st.title("Clasificador de imágenes con Streamlit")

uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen cargada', use_column_width=True)
    st.write("")
    st.write("Clasificando...")
    res = predict(image,model)
    st.write(f"Predicción: {res}")



