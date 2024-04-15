
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import os
import torch
from cnn import SimpleCNN,CNN
from torchvision import transforms
import torchvision
from PIL import Image
import numpy as np

img_size = 224

def load_model():
   dict = torch.load("DeepLearning/model_48.pth",map_location=torch.device('cpu'))
   dict = torch.load("DeepLearning/modelo_final.pth")
   # Cargar el modelo
   num_classes = 15
   list_dropouts =  [0.4154439238682205,0.21085094262884393,0.4313465383566946, 0.32912701708030556,0.4898772544775635]
   list_neuronas = [199,194,160,114,171]
   modelo = SimpleCNN(torchvision.models.resnet50(), n_classes=num_classes,n_layers=5, unfreezed_layers=7,list_dropouts=list_dropouts,list_neuronas_salida=list_neuronas)

   modelo.load_state_dict(dict)
   modelo.eval()
   return modelo

model = load_model()


class_to_index = {0: 'Inside city', 1: 'Kitchen', 2: 'Office', 3: 'Store', 4: 'Street', 5: 'Suburb', 6: 'Highway', 7: 'Coast', 8: 'Mountain', 9: 'Open country', 10: 'Industrial', 11: 'Forest', 12: 'Tall building', 13: 'Bedroom', 14: 'Living room'}
errors_by_class = {class_name: 0 for class_name in class_to_index.values()}
elements_by_class = {class_name: 0 for class_name in class_to_index.values()}


# Configura tu dataset de validación y DataLoader aquí
# Por ejemplo:
transformaciones = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        transforms.Resize(size =(img_size,img_size)),
    ])


    
carpeta_origen = "DeepLearning/DataAugmentated"
#carpeta_origen = "DeepLearning/datasetcompleto"


# Evalúa el modelo
# Cargar las imágenes utilizando ImageFolder
archivos = [f for f in os.listdir(carpeta_origen) if os.path.isfile(os.path.join(carpeta_origen, f))]

errores =0 
# Iterar sobre el dataset transformado y guardar las imágenes
for i,archivo in enumerate(archivos):
    imagen_path = os.path.join(carpeta_origen, archivo)
    imagen = Image.open(imagen_path)
    #print("Se ha cargado {}".format(imagen_path))
    class_name = archivo.split('_')[0]

    # Aplicar transformaciones de nuevo a la misma imagen
    input_tensor = transformaciones(imagen)
    
    input_tensor = input_tensor.unsqueeze(0)

    outputs = model(input_tensor)
    salida = class_to_index[outputs.argmax(1).item()]
    elements_by_class[class_name] = elements_by_class[class_name]+1
    print("La clase de entrada era: {}, la predicha es: {}".format(class_name,salida))
    if salida== class_name:
        # Se actualiza el diccionario de errores y se pone un error 
        pass
    else:
        errors_by_class[class_name] = errors_by_class[class_name]+1
        errores = errores+1
result_dict = {key: (elements_by_class[key] -errors_by_class[key])*100 / elements_by_class[key] for key in errors_by_class if key in elements_by_class}
print("La precisión final es de: {}".format((len(archivos)-errores)/len(archivos)))

plt.figure(figsize=(10, 6))
plt.bar(result_dict.keys(), result_dict.values(), color='skyblue')
plt.xlabel('Clase')
plt.ylabel('Valor')
plt.title('Valor por Clase')
plt.xticks(rotation=45)
plt.show()