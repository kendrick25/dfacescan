import cv2
import numpy as np
import os

# Configuración de rutas
input_image_path = r'D:\2024\IA de Reconocimiento\dfacescan\ImagenesPrueba\img2.jpg'
output_directory = r'D:\2024\IA de Reconocimiento\dfacescan\PruebasLibrerias\ImagenesProcesadas'

# Crear directorio de salida si no existe
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Cargar el clasificador preentrenado de rostros de OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Cargar la imagen
image = cv2.imread(input_image_path)
height, width = image.shape[:2]

# Detección de rostros en la imagen completa
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces_full_image = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Dibujar recuadros en los rostros detectados en la imagen completa
for (x, y, w, h) in faces_full_image:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Guardar la imagen completa con los recuadros dibujados
full_image_filename = os.path.join(output_directory, 'full_image_with_boxes.jpg')
cv2.imwrite(full_image_filename, image)

# Definir las secciones de la imagen (4 secciones)
sections = [
    image[0:height//2, 0:width//2],       # Sección superior izquierda
    image[0:height//2, width//2:width],   # Sección superior derecha
    image[height//2:height, 0:width//2],  # Sección inferior izquierda
    image[height//2:height, width//2:width]  # Sección inferior derecha
]

# Procesar cada sección
processed_sections = []
for i, section in enumerate(sections):
    # Hacer zoom en la sección
    zoom_factor = 1.5
    zoomed_section = cv2.resize(section, None, fx=zoom_factor, fy=zoom_factor)

    # Convertir a escala de grises para detección de rostros
    gray_zoomed = cv2.cvtColor(zoomed_section, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en la sección ampliada con zoom
    faces = face_cascade.detectMultiScale(gray_zoomed, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Dibujar recuadros en los rostros detectados
    for (x, y, w, h) in faces:
        cv2.rectangle(zoomed_section, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Guardar la sección ampliada con zoom y recuadros dibujados
    zoomed_section_filename = os.path.join(output_directory, f'section_zoom_{i+1}.jpg')
    cv2.imwrite(zoomed_section_filename, zoomed_section)
    
    # Redimensionar la sección procesada a su tamaño original
    processed_section = cv2.resize(zoomed_section, (section.shape[1], section.shape[0]))

    # Guardar la sección procesada en la lista
    processed_sections.append(processed_section)

# Unir las secciones en una sola imagen
top_half = np.hstack((processed_sections[0], processed_sections[1]))
bottom_half = np.hstack((processed_sections[2], processed_sections[3]))
final_image = np.vstack((top_half, bottom_half))

# Guardar la imagen final con los recuadros dibujados en las secciones
final_image_filename = os.path.join(output_directory, 'final_combined_image_with_boxes.jpg')
cv2.imwrite(final_image_filename, final_image)

print(f"Imagen completa con recuadros guardada en: {full_image_filename}")
print(f"Imagen final combinada con recuadros guardada en: {final_image_filename}")
