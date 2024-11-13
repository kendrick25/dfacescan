import cv2  
import os
import imutils
from deepface import DeepFace

dataPath = 'data'  # Carpeta para almacenar los rostros procesados
personsPath = 'persons'  # Carpeta donde se almacenan las imágenes de las personas
margin_ratio = 0  # Margen adicional para el área de recorte

# Iterar sobre cada persona en la carpeta 'persons'
for personName in os.listdir(personsPath):
    imagesPath = os.path.join(personsPath, personName)
    personDataPath = os.path.join(dataPath, personName)
    
    # Crear carpeta para la persona en 'data' si no existe
    if not os.path.exists(personDataPath):
        print('Carpeta Creada:', personDataPath)
        os.makedirs(personDataPath)
    
    # Obtener todas las imágenes de la persona
    images = [f for f in os.listdir(imagesPath) if os.path.isfile(os.path.join(imagesPath, f))]
    count = 0
    max_resize_attempts = 5  # Máximo número de intentos de redimensionamiento

    # Procesar cada imagen
    for image_name in images:
        image_path = os.path.join(imagesPath, image_name)
        frame = cv2.imread(image_path)
        
        if frame is None:
            print(f"No se pudo leer la imagen {image_name}")
            continue

        resize_attempts = 0
        faces_detected = False

        # Intentar redimensionar hasta detectar rostros
        while resize_attempts < max_resize_attempts and not faces_detected:
            frame_resized = imutils.resize(frame, width=320 + resize_attempts * 100)
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            scale_factor = frame.shape[1] / frame_resized.shape[1]
            
            # Usar DeepFace para detectar rostros con RetinaFace
            try:
                faces = DeepFace.extract_faces(img_path=frame_resized, detector_backend='retinaface', enforce_detection=True)
                faces_detected = True
                
                for face_data in faces:
                    facial_area = face_data['facial_area']  # Obtener el área facial
                    x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                    
                    # Calcular margen adicional
                    margin_x = int(w * margin_ratio)
                    margin_y = int(h * margin_ratio)
                    
                    # Ajustar coordenadas al tamaño original y agregar margen
                    x = int(x * scale_factor) - margin_x
                    y = int(y * scale_factor) - margin_y
                    w = int(w * scale_factor) + 2 * margin_x
                    h = int(h * scale_factor) + 2 * margin_y

                    # Recortar área ampliada del rostro
                    rostro = frame[max(y, 0):min(y + h, frame.shape[0]), max(x, 0):min(x + w, frame.shape[1])]

                    # Guardar el rostro etiquetado en la carpeta de la persona
                    cv2.imwrite(os.path.join(personDataPath, f'{personName}_rostro_{count}.jpg'), rostro)
                    count += 1
            except Exception as e:
                print(f"No se detectaron rostros en la imagen {image_name}, intento {resize_attempts + 1}: {str(e)}")

            resize_attempts += 1

        if not faces_detected:
            print(f"No se detectaron rostros después de {max_resize_attempts} intentos en la imagen {image_name}.")

print("Proceso de detección y almacenamiento de rostros completado.")
