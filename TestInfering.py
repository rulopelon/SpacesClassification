import torch
import torchvision
from torchvision import transforms
from PIL import Image
from  torchvision.models import resnet50
import torch
from cnn import SimpleCNN,CNN
from torchvision import transforms
from PIL import Image

dict = torch.load("model_48.pth",map_location=torch.device('cpu'))
# Carga el modelo
num_classes = 15
list_dropouts =  [0.4154439238682205,0.21085094262884393,0.4313465383566946, 0.32912701708030556,0.4898772544775635]
list_neuronas = [199,194,160,114,171]
modelo = SimpleCNN(torchvision.models.resnet50(), n_classes=num_classes,n_layers=5, unfreezed_layers=7,list_dropouts=list_dropouts,list_neuronas_salida=list_neuronas)

modelo.load_state_dict(dict)
modelo.eval()  # Cambia el modelo a modo de evaluación

img_size = 224
# Define image transformations
transformaciones = transforms.Compose([
        transforms.ToTensor(),
    transforms.Resize(size =(img_size,img_size)),
    #transforms.Normalize(mean=[0.485], std=[0.229])
    # Añade aquí más transformaciones según sea necesario
])


input_image = Image.open('image_0114.jpg')
input_tensor = transformaciones(input_image)# Add batch dimension
outputs = modelo(input_tensor)
print(outputs)