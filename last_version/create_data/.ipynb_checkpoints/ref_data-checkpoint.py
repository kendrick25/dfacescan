import cv2   
import os
from deepface import DeepFace

dataPath = 'ref-standard'  # Carpeta para almacenar los rostros procesados
personsPath = 'ref'  # Carpeta donde se almacenan las imágenes de las personas
margin_ratio = 0.2  # Margen adicional para el área de recorte
output_width = 224  # Ancho deseado para la imagen de recorte
aspect_ratio = 3 / 4  # Relación de aspecto 4:3

# Calcular la altura basada en la relación de aspecto
output_height = int(output_width / aspect_ratio)

# Iterar sobre cada persona en la carpeta 'persons'
for personName in os.listdir(personsPath):
    imagesPath = os.path.join(personsPath, personName)
    personDataPath = os.path.join(dataPath, personName)
    
    # Crear carpeta para la persona en 'data' si no existe
    if not os.path.exists(personDataPath):
        print('Carpeta creada:', personDataPath)
        os.makedirs(personDataPath)
    
    # Obtener todas las imágenes de la persona
    images = [f for f in os.listdir(imagesPath) if os.path.isfile(os.path.join(imagesPath, f))]
    count = 0  # Contador para los nombres de archivo
    
    # Procesar cada imagen
    for image_name in images:
        image_path = os.path.join(imagesPath, image_name)
        frame = cv2.imread(image_path)
        
        if frame is None:
            print(f"No se pudo leer la imagen {image_name}")
            continue

        # Usar DeepFace para detectar rostros con RetinaFace
        try:
            faces = DeepFace.extract_faces(img_path=frame, detector_backend='retinaface', enforce_detection=False)
            
            if faces:
                for face_data in faces:
                    facial_area = face_data['facial_area']  # Obtener el área facial
                    x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                    
                    # Calcular margen adicional
                    margin_x = int(w * margin_ratio)
                    margin_y = int(h * margin_ratio)
                    
                    # Ajustar coordenadas para incluir el margen
                    x = max(x - margin_x, 0)
                    y = max(y - margin_y, 0)
                    w = min(w + 2 * margin_x, frame.shape[1] - x)
                    h = min(h + 2 * margin_y, frame.shape[0] - y)

                    # Recortar área ampliada del rostro
                    rostro = frame[y:y + h, x:x + w]

                    # Verificar si el recorte está vacío
                    if rostro.size == 0:
                        print(f"Rostro vacío en la imagen {image_name}")
                        continue

                    # Redimensionar el rostro recortado para que tenga un tamaño fijo con la relación 4:3
                    rostro_resized = cv2.resize(rostro, (output_width, output_height))

                    # Obtener la extensión original para guardar con el mismo formato
                    ext = os.path.splitext(image_name)[1].lower()

                    # Asignar un nombre secuencial 'img-XX.ext'
                    output_filename = f"img-{count:02d}{ext}"
                    output_path = os.path.join(personDataPath, output_filename)
                    
                    # Guardar la imagen redimensionada con calidad ajustada según el formato
                    if ext in ['.jpg', '.jpeg']:
                        cv2.imwrite(output_path, rostro_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    elif ext == '.png':
                        cv2.imwrite(output_path, rostro_resized, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                    else:
                        cv2.imwrite(output_path, rostro_resized)

                    print(f"Imagen guardada: {output_filename}")
                    
                    count += 1  # Incrementar el contador para el próximo nombre

            else:
                print(f"No se detectaron rostros en la imagen {image_name}")
                
        except Exception as e:
            print(f"Error al procesar la imagen {image_name}: {str(e)}")

print("Proceso de detección y almacenamiento de rostros completado.")
