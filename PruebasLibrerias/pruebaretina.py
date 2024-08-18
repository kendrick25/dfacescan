import cv2
import numpy as np
import os
from retinaface import RetinaFace

# Configuración de rutas
input_image_path = r'C:\Users\kendr\OneDrive\Escritorio\ProeyectoIA\dfacescan\ImagenesPrueba\img2.jpg'
output_directory = r'C:\Users\kendr\OneDrive\Escritorio\ProeyectoIA\dfacescan\PruebasLibrerias\ImagenesProcesadas'

# Crear directorio de salida si no existe
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Cargar la imagen
image = cv2.imread(input_image_path)
height, width = image.shape[:2]

# Detección de rostros en la imagen completa
faces_full_image = RetinaFace.detect_faces(input_image_path)

# Dibujar recuadros en los rostros detectados en la imagen completa
for key in faces_full_image.keys():
    face = faces_full_image[key]
    facial_area = face["facial_area"]
    cv2.rectangle(image, (facial_area[0], facial_area[1]), (facial_area[2], facial_area[3]), (255, 0, 0), 2)

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

    # Guardar la sección ampliada con zoom antes de la detección de rostros
    zoomed_section_filename_before = os.path.join(output_directory, f'section_zoom_{i+1}_before_detection.jpg')
    cv2.imwrite(zoomed_section_filename_before, zoomed_section)

    # Detectar rostros en la sección ampliada con zoom usando RetinaFace
    faces = RetinaFace.detect_faces(zoomed_section_filename_before)

    # Dibujar recuadros en los rostros detectados
    for key in faces.keys():
        face = faces[key]
        facial_area = face["facial_area"]
        cv2.rectangle(zoomed_section, (facial_area[0], facial_area[1]), (facial_area[2], facial_area[3]), (255, 0, 0), 2)

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
