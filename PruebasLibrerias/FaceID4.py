import cv2
import numpy as np
import os

# Configuración de rutas
input_image_path = r'D:\Nueva carpeta\Prueba\dfacescan\ImagenesPrueba\img2.jpg'
output_directory = r'D:\Nueva carpeta\Prueba\dfacescan\PruebasLibrerias\ImagenProcesada'
#output_directory = r'D:\Nueva carpeta\Prueba\IA-de-reconocimiento\PruebasLibrerias\ImagenesProcesadas'

# Crear directorio de salida si no existe
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Cargar el clasificador preentrenado de rostros de OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Cargar la imagen
image = cv2.imread(input_image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detectar rostros
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Dibujar recuadros y guardar imágenes
for i, (x, y, w, h) in enumerate(faces):
    # Dibujar un recuadro alrededor del rostro
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Guardar la imagen con recuadros
boxed_image_filename = os.path.join(output_directory, 'boxed_image.jpg')
cv2.imwrite(boxed_image_filename, image)

print(f"Imagen con recuadros guardada en: {boxed_image_filename}")