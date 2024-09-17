# Prueba en Google Colab
## Montar Google Drive en Colab

```python
from google.colab import drive
drive.mount('/content/drive')
```
> Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).

## 1. Verificar la versión de Python

```bash
!python --version
```
> Python 3.10.12

## 2. Instalar PyTorch, torchvision, torchaudio y cudatoolkit

```bash
!pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```
> Looking in indexes: https://pypi.org/simple, https://download.pytorch.org/whl/cu118  
> Requirement already satisfied: torch in /usr/local/lib/python3.10/dist-packages (2.4.0+cu121)  
> Requirement already satisfied: torchvision in /usr/local/lib/python3.10/dist-packages (0.19.0+cu121)  
> Requirement already satisfied: torchaudio in /usr/local/lib/python3.10/dist-packages (2.4.0+cu121)  
> Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from torch) (3.16.0)  
> Requirement already satisfied: typing-extensions>=4.8.0 in /usr/local/lib/python3.10/dist-packages (from torch) (4.12.2)  
> Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch) (1.13.2)  
> Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch) (3.3)  
> Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch) (3.1.4)  
> Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from torch) (2024.6.1)  
> Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from torchvision) (1.26.4)  
> Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /usr/local/lib/python3.10/dist-packages (from torchvision) (9.4.0)  
> Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch) (2.1.5)  
> Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.10/dist-packages (from sympy->torch) (1.3.0)  

## 3. Instalar OpenCV

```bash
!pip install opencv-python
```
> Requirement already satisfied: opencv-python in /usr/local/lib/python3.10/dist-packages (4.10.0.84)  
> Requirement already satisfied: numpy>=1.21.2 in /usr/local/lib/python3.10/dist-packages (from opencv-python) (1.26.4)  

## 4. Instalar el repositorio DSFD-Pytorch-Inference

```bash
!pip install git+https://github.com/hukkelas/DSFD-Pytorch-Inference.git
```
> Collecting git+https://github.com/hukkelas/DSFD-Pytorch-Inference.git  
> Cloning https://github.com/hukkelas/DSFD-Pytorch-Inference.git to /tmp/pip-req-build-5xuuyquj  
> Running command git clone --filter=blob:none --quiet https://github.com/hukkelas/DSFD-Pytorch-Inference.git /tmp/pip-req-build-5xuuyquj  
> Resolved https://github.com/hukkelas/DSFD-Pytorch-Inference.git to commit dde9c7dd9cdc9254c2ca345222c86a8ecfa17f5b  
> Preparing metadata (setup.py) ... done  
> Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from face_detection==0.2.1) (1.26.4)  
> Building wheels for collected packages: face_detection  
> Building wheel for face_detection (setup.py) ... done  
> Created wheel for face_detection: filename=face_detection-0.2.1-py3-none-any.whl size=29972 sha256=51c177cb133e2dec232167aaeeaa5d78f556294c19a2b009653bd5e94674d860  
> Stored in directory: /tmp/pip-ephem-wheel-cache-ettgqy7k/wheels/31/fc/c5/28af01da09c7625bd966f9871b081cb72e131ffb926c0de66b  
> Successfully built face_detection  
> Installing collected packages: face_detection  
> Successfully installed face_detection-0.2.1  

## 5. Preparar el directorio para las imágenes

```python
import os

# Ruta de almacenamiento en Google Drive
save_path = '/content/drive/MyDrive/PruebaU/Imagenes'

# Verificar si la ruta existe, si no, la crea
if not os.path.exists(save_path):
    os.makedirs(save_path)

print(f"Imágenes procesadas se guardarán en: {save_path}")
```
> Imágenes procesadas se guardarán en: /content/drive/MyDrive/PruebaU/Imagenes

## 6. Prueba del codigo test.py

```python
import os
import cv2
import time
import face_detection

def draw_faces(im, bboxes):
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 2)

# Ruta de la imagen de entrada
input_image_path = "/content/drive/MyDrive/PruebaU/Imagenes/img2.jpg"

# Inicializar el detector
detector = face_detection.build_detector(
    "DSFDDetector",
    max_resolution=1080
)

# Leer la imagen
im = cv2.imread(input_image_path)
if im is None:
    print(f"Error: la imagen no se pudo cargar desde {input_image_path}")
else:
    print(f"Processing: {input_image_path}")
    t = time.time()

    # Detectar rostros
    dets = detector.detect(
        im[:, :, ::-1]
    )[:, :4]

    print(f"Detection time: {time.time() - t:.3f}")

    # Dibujar los rostros detectados
    draw_faces(im, dets)

    # Generar la ruta de salida (mismo nombre con "_out" añadido)
    output_image_path = input_image_path.replace(".jpg", "_out.jpg")

    # Guardar la imagen procesada
    cv2.imwrite(output_image_path, im)
    print(f"Imagen guardada en: {output_image_path}")
```
> Processing: /content/drive/MyDrive/PruebaU/Imagenes/img2.jpg  
> Detection time: 20.707  
> Imagen guardada en: /content/drive/MyDrive/PruebaU/Imagenes/img2_out.jpg

## 7. Verificar las imágenes Procesadas

```python
import cv2
import matplotlib.pyplot as plt

# Cargar y mostrar una imagen procesada
img = cv2.imread('/content/drive/MyDrive/PruebaU/Imagenes/img2_out.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
```
```