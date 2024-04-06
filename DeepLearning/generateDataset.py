import os
import shutil

# Define el directorio de origen y el directorio de destino
directorio_origen = 'DeepLearning/dataset'
directorio_destino = 'DeepLearning/datasetcompleto'
archivo_registro = 'registro_carpetas.txt'

clases = []
# Asegúrate de que el directorio destino existe, si no, créalo
os.makedirs(directorio_destino, exist_ok=True)

# Itera sobre todos los directorios y subdirectorios en el directorio origen
with open(archivo_registro, 'a') as registro:
    for raiz, carpetas, archivos in os.walk(directorio_origen):
        for archivo in archivos:
            # Obtén la ruta completa del archivo original
            ruta_completa = os.path.join(raiz, archivo)
            
            # Extrae el nombre de la carpeta original
            nombre_carpeta = os.path.basename(raiz)
            if nombre_carpeta not in clases:
                clases.append(nombre_carpeta)
                registro.write(nombre_carpeta + '\n')

            # Crea el nuevo nombre del archivo incluyendo el nombre de la carpeta original
            nuevo_nombre = f"{nombre_carpeta}_{archivo}"
            
            # Define la ruta completa del nuevo archivo en el directorio destino
            ruta_destino = os.path.join(directorio_destino, nuevo_nombre)
            
            # Copia el archivo al directorio destino con el nuevo nombre
            shutil.copy(ruta_completa, ruta_destino)
            
            print(f"Copiado: {ruta_completa} a {ruta_destino}")