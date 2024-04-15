import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Dataset
import numpy as np
import scipy
import os
from tqdm import tqdm
import time
from PIL import Image
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

directorioTrain = "datasetAugmented/training"
directorioTest = "datasetAugmented/validation"
archivo_registro = 'registro_carpetas.txt'

class CustomDataset(Dataset):
    """Dataset para cargar archivos .jpg."""
    
    def __init__(self, directory, classes,transform=None):
        """
        Args:
            directory (string): Directorio con todos los archivos .mat.
            transform (callable, optional): Opcional transformación a ser aplicada
                en una muestra.
        """
        self.directory = directory
        self.transform = transform
        self.files = [f for f in os.listdir(directory) if f.endswith('.jpg')]
        self.filenames = os.listdir(directory)
        self.classes = classes
        self.class_to_index = {cls: idx for idx, cls in enumerate(classes)}

        
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        filename = self.filenames[idx]

        class_name = filename.split('_')[0]
        
        # Convertir la etiqueta de clase a un índice
        class_idx = self.class_to_index[class_name]
       
        label_one_hot = torch.zeros(len(self.classes))
        label_one_hot[class_idx] = 1
        image = Image.open(os.path.join(self.directory,filename))
        convertir_a_tensor =  transforms.Compose([
    #transforms.Grayscale(num_output_channels=1), # Convertir a escala de grises
    transforms.Resize(size = (224,224)),
    transforms.ToTensor() # Convertir a tensor
])

        image = convertir_a_tensor(image)
        return image,label_one_hot



class EarlyStopper:
    def __init__(self, patience=4, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        # TODO: La epoch anterior menos la actual tien que ser menor que el delta 0.0001 paciencia 0
        return False
    
class SimpleCNN(nn.Module):
    def __init__(self,base_model,n_layers,n_classes,unfreezed_layers,list_dropouts,list_neuronas_salida):
        super(SimpleCNN, self).__init__()
        self.earlyStopper = EarlyStopper()
        self.base_model = base_model
        self.num_classes = n_classes
        self.n_layers = n_layers
        self.in_features = self.base_model.fc.in_features
        # Freeze convolutional layers
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Unfreeze specified number of layers
        if unfreezed_layers > 0:
            for layer in list(self.base_model.children())[-unfreezed_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
        for i in range(1, self.n_layers-1):
            if i== 1:
                setattr(self, f'fc{i}', nn.Sequential(
                    nn.Dropout(list_dropouts[i]),
                    nn.Linear(self.base_model.fc.in_features, list_neuronas_salida[i]),
                    nn.ReLU()))
            else:
                setattr(self, f'fc{i}', nn.Sequential(
                    nn.Dropout(list_dropouts[i]),
                    nn.Linear( list_neuronas_salida[i-1], list_neuronas_salida[i]),
                    nn.ReLU()))
        # Add a new softmax output layer
        self.fc_final = nn.Sequential(

            nn.Dropout(list_dropouts[-1]),
            nn.Linear(list_neuronas_salida[-2], self.num_classes),
            nn.Softmax(dim=1)
        )

        # Replace the last layer of the base model
        self.base_model.fc = nn.Identity()
    

    def forward(self, x):

        x = self.base_model(x)


            
        x = x.view(-1,self.in_features)
        for i in range(1, self.n_layers-1):  # Asume que 'channel_list' tiene la cantidad correcta de canales.
            fc = getattr(self, f'fc{i}')
            x = fc(x)
        x = self.fc_final(x)

        return x
    
    
def train_and_validate(model, device, train_loader, val_loader, epochs, optimizer):
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'train_accuracy': [], 'valid_loss': [], 'valid_accuracy': []}
    print("Empieza el entrenamiento")

    creado = False
    for epoch in tqdm(range(epochs)):
        model.train()
        training_losses = []
        tiempo_comienzo = time.time()
        correct_training = 0
        train_loss = 0.0
        train_accuracy = 0.0

        tiempo_inicial = time.time()
        for images, labels in train_loader:
            tiempo_actual = time.time()
            tiempo_inicial =tiempo_actual
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            try:
                outputs = model(images)
            except Exception as e:
                print(e)
                torch.cuda.empty_cache()
                print("Combinación incorrecta")

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_accuracy += (outputs.argmax(1) == labels.argmax(1)).sum().item()

        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader.dataset)
        train_accuracy = train_accuracy*100
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)

        print(f'Epoch {epoch + 1}/{epochs} - '
                f'Train Loss: {train_loss:.4f}, '
                f'Train Accuracy: {train_accuracy:.4f}')

        model.eval()
        valid_loss = 0
        valid_accuracy = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            valid_accuracy += (outputs.argmax(1) == labels.argmax(1)).sum().item()

        valid_loss /= len(val_loader)
        valid_accuracy /= len(val_loader.dataset)
        valid_accuracy = valid_accuracy*100
        history['valid_loss'].append(valid_loss)
        history['valid_accuracy'].append(valid_accuracy)

        print(f'Epoch {epoch + 1}/{epochs} - '
                f'Validation Loss: {valid_loss:.4f}, '
                f'Validation Accuracy: {valid_accuracy:.4f}')
       


        model_file = "modelo_final.pth"
        if model.earlyStopper.early_stop(valid_loss):
            print("Se ha hecho early stopping")
            
            torch.save(model.state_dict(), model_file)
            #wandb.save(model_file)
            return valid_accuracy
            #validation_accuracy = input("Dime la precisión")
            #trial.report(validation_accuracy,epoch+1)
            #raise optuna.exceptions.TrialPruned()
        
    # UNa vez que ha terminado de entrenar, lo guarda
    model_file = "modelo_final.pth"
    torch.save(model.state_dict(), model_file)
    #wandb.save(model_file)

    return valid_accuracy

def entrenar():


    with open(archivo_registro, 'r') as archivo:
        clases = archivo.readlines()

    # Remover los saltos de línea y posibles duplicados
    clases = list(set([nombre_clase.strip() for nombre_clase in clases]))
    n_classes = len(clases)
    # Hiperparámetros a optimizar
    n_layers =5
    lr =1.0222409633565202e-05
    optimizer_name = "Adam"

    unfreezed_layers = 7
    # Eligiendo el número de capas internas
    list_neuronas_salida = [199,194,160,114,171]
    list_dropouts = [0.4154439238682205,0.21085094262884393,0.4313465383566946,0.32912701708030556,0.4898772544775635]

    base_model = torchvision.models.resnet50(weights='DEFAULT')



    try:
        model = SimpleCNN(base_model,n_layers,n_classes,unfreezed_layers,list_dropouts,list_neuronas_salida).to(device)
        model = model.to(device)
    except Exception as e:
        print(e)

        print("Demasiado grande")
        torch.cuda.empty_cache()
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    


    with open(archivo_registro, 'r') as archivo:
        clases = archivo.readlines()

    # Remover los saltos de línea y posibles duplicados
    clases = list(set([nombre_clase.strip() for nombre_clase in clases]))
    train_dataset = CustomDataset(directory=directorioTrain,classes=clases)
    test_dataset = CustomDataset(directory=directorioTest,classes=clases)

    # Datos de entrenamiento y validación
    batch_size = 128 #TODO Aumentar batch size 128 meter en optuna
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size, shuffle=True,num_workers=2)

    val_loader = DataLoader(test_dataset,
                            batch_size=batch_size, shuffle=False,num_workers=2)

    epochs = 1000  # Puedes ajustar esto según sea necesario
   
    accuracy = train_and_validate(model, device, train_loader, val_loader, epochs, optimizer)

    return accuracy


                                  
if __name__ == '__main__':
    #torch.multiprocessing.set_start_method("spawn")# good solution !!!!
    accuracy = entrenar()
    print(accuracy)








