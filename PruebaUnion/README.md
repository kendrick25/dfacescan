# Proyecto de Detección y Entrenamiento de Rostros

Este proyecto se enfoca en la detección de rostros a partir de imágenes y el entrenamiento de un modelo de reconocimiento facial. Utiliza la librería OpenCV para la manipulación de imágenes y la detección de rostros, con un enfoque específico en la recolección de datos y el preprocesamiento de imágenes. Tambien se busca combinar el uso de las librerias ya definidas para la deteccion de rostros.

## Archivos Incluidos

### 1. `Base_Data_Image.py`
Este script se encarga de:
- Detectar rostros en una carpeta de imágenes proporcionada.
- Redimensionar las imágenes varias veces hasta que se detecte al menos un rostro o se alcance un número máximo de intentos.
- Guardar las imágenes de los rostros detectados en una carpeta específica para su posterior uso en entrenamiento.

#### Proceso:
- Se especifica el nombre de la persona (`personName`) y las rutas de las imágenes y la carpeta de almacenamiento de rostros.
- Si no se detecta un rostro en la imagen, el script intenta redimensionarla varias veces (hasta 5 intentos) incrementando su tamaño en cada intento.
- Cuando se detecta un rostro, se recorta y redimensiona a 720x720 píxeles y se guarda en la carpeta de destino.

### 2. `Entrenamiento.py`
Este script se encarga de:
- Leer las imágenes de rostros previamente almacenadas en diferentes carpetas (una por cada persona).
- Preparar los datos (imágenes y etiquetas) para el entrenamiento de un modelo de reconocimiento facial.

#### Proceso:
- Se especifica la ruta donde se encuentran las carpetas con los rostros de diferentes personas (`dataPath`).
- Se recorren todas las imágenes dentro de las carpetas de personas, y se almacenan en una lista las imágenes junto con sus etiquetas correspondientes.
- Estas imágenes se usarán para entrenar un modelo de reconocimiento facial en pasos futuros.


markdown
Copiar código

## Dependencias

Este proyecto requiere las siguientes dependencias:
- pip install imutils
- pip install face-recognition
- pip install --upgrade opencv-python

Para instalarlas, puedes usar pip:
```bash
pip install opencv-python imutils
```
Uso
1. Detección y almacenamiento de rostros
Ejecuta el script Base_Data_Image.py para detectar rostros en las imágenes de una carpeta y almacenarlos en la carpeta correspondiente para cada persona.

```bash
python Base_Data_Image.py
```
2. Preparación para el entrenamiento
Ejecuta el script Entrenamiento.py para leer las imágenes de los rostros y preparar los datos para el entrenamiento.

```bash
python Entrenamiento.py
```
