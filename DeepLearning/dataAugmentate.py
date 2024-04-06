import os
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from PIL import Image
import numpy as np

# Definir la ruta de la carpeta de origen y la carpeta destino
carpeta_origen = 'DeepLearning/datasetcompleto'
carpeta_destino_train = 'DeepLearning/datasetAugmented/training'
carpeta_destino_test = 'DeepLearning/datasetAugmented/validation'

# Asegurarse de que el directorio destino existe, si no, crearlo
os.makedirs(carpeta_destino_test, exist_ok=True)
os.makedirs(carpeta_destino_train, exist_ok=True)

img_size = 224
num_imagenes_por_original = 20

# Definir las transformaciones para data augmentation
transformaciones = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(img_size),
    transforms.RandomResizedCrop(size=(img_size, img_size), antialias=True),
    transforms.GaussianBlur(kernel_size=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    #transforms.Normalize(mean=[0.485], std=[0.229])
    # Añade aquí más transformaciones según sea necesario
])

# Cargar las imágenes utilizando ImageFolder
archivos = [f for f in os.listdir(carpeta_origen) if os.path.isfile(os.path.join(carpeta_origen, f))]

# Iterar sobre el dataset transformado y guardar las imágenes
for i,archivo in enumerate(archivos):
    imagen_path = os.path.join(carpeta_origen, archivo)
    imagen = Image.open(imagen_path)
    print("Se ha cargado {}".format(imagen_path))
    for j in range(num_imagenes_por_original):

        # Aplicar transformaciones de nuevo a la misma imagen
        imagen_transformada = transformaciones(imagen)
        # Definir la ruta de archivo donde se guardará la imagen
        es_train = np.random.rand() < 0.8
        carpeta_destino = carpeta_destino_train if es_train else carpeta_destino_test
        
        # Definir la ruta de archivo para guardar la imagen transformada
        ruta_archivo = os.path.join(carpeta_destino, f'{os.path.splitext(archivo)[0]}-transformada_{j}-{os.path.splitext(archivo)[1]}')
        
        # Guardar la imagen transformada
        save_image(imagen_transformada, ruta_archivo)

print(f'Imágenes transformadas guardadas en {carpeta_destino}')