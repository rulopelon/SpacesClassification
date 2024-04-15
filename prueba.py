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

import streamlit as st
from PIL import Image
import torch
from cnn import SimpleCNN
import torchvision
from torchvision import transforms

# Funciones auxiliares (load_model, predict)...

# Configuración de la página
st.set_page_config(
    page_title="Clasificador de Imágenes",
    page_icon="📷",
    layout="wide"
)

# Aplicar tema personalizado (puedes editar los colores)
st.markdown(
    """
    <style>
    .main {
    background-color: #f0f2f6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Cargar el modelo
model = load_model()

titulo_imagen_path = 'logo-icai.png' 
titulo_imagen = Image.open(titulo_imagen_path)

tamaño_imagen_titulo = 300
tamaño_imagen_titulo_2 = 150


titulo_imagen_path_2 = 'logo_equipo.webp'  
titulo_imagen_2 = Image.open(titulo_imagen_path_2)

# Colocar la imagen del título en la página
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.image(titulo_imagen_2, width=tamaño_imagen_titulo_2)

with col3:
    st.image(titulo_imagen, width=tamaño_imagen_titulo)

st.title("🖼️ Clasificador de Imágenes")

# Columnas para organizar el contenido
col4, col5 = st.columns(2)

with col4:
    opcion_seleccionada = st.selectbox(
        'Seleccione el tipo de imagen que va a introducir:',
        ('Inside city', 'Kitchen', 'Office', 'Store', 'Street', 'Suburb', 'Highway', 'Coast', 'Mountain', 'Open country', 'Industrial', 'Forest', 'Tall building', 'Bedroom', 'Living room')
    )



with col5:
    uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "png"], help='Arrastra una imagen o haz clic para seleccionarla')

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 3])
    with col4:
        st.image(image, caption='Imagen cargada', use_column_width=True)
    with col5:
        st.write("Clasificando...")
        res = predict(image, model)
        # Verificar si la opción seleccionada coincide con la predicción del modelo
        if opcion_seleccionada == res:
            st.success("¡Correcto!")
            st.write(f"Predicción: {res} | imagen introducida: {opcion_seleccionada}", unsafe_allow_html=True)
        else:
            st.error("Incorrecto")
            st.write(f"Predicción: {res} | imagen introducida: {opcion_seleccionada}", unsafe_allow_html=True)

