import os
import cv2
import joblib
import numpy as np
from deepface import DeepFace

# Rutas de entrada y salida
test_path = "./test"  # Carpeta con imágenes de entrada
result_path = "result"  # Carpeta para almacenar la imagen con rectángulos y etiquetas
model_path = './models/face_classifier.pkl'  # Ruta del modelo clasificador
encoder_path = './models/label_encoder.pkl'  # Ruta del codificador de etiquetas

# Crear carpeta para almacenar la imagen resultante si no existe
os.makedirs(result_path, exist_ok=True)

# Cargar el clasificador y el codificador de etiquetas
classifier = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)

# Obtener todas las imágenes en la carpeta de prueba
image_files = [f for f in os.listdir(test_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Lista de detectores para probar
detectors = ['retinaface', 'mtcnn', 'opencv', 'ssd', 'dlib']

# Definir un tamaño mínimo para los rostros (ajusta según sea necesario)
MIN_FACE_WIDTH = 30
MIN_FACE_HEIGHT = 30

def process_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"No se pudo leer la imagen: {image_path}")
        return None
    return frame

def detect_faces(frame, img_file):
    faces = None
    for detector in detectors:
        try:
            # Intentar extraer rostros con diferentes detectores
            faces = DeepFace.extract_faces(img_path=image_path, detector_backend=detector, enforce_detection=False)
            if faces:
                print(f"Rostros detectados usando el detector: {detector}")
                break  # Si detecta rostros, salir del bucle
        except Exception as e:
            print(f"Error con el detector {detector} en {img_file}: {str(e)}")
    return faces

def process_faces(frame, faces, img_file):
    detecciones = {}
    for count, face_data in enumerate(faces):
        facial_area = face_data['facial_area']  # Obtener el área facial
        x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']

        # Verificar que las dimensiones sean válidas y recortar el área del rostro
        if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
            print(f"Rostro {count} en {img_file} está fuera de los límites de la imagen.")
            continue

        # Verificar tamaño mínimo para evitar errores
        if w < MIN_FACE_WIDTH or h < MIN_FACE_HEIGHT:
            print(f"Rostro {count} en {img_file} es demasiado pequeño para ser considerado válido.")
            continue

        rostro = frame[y:y + h, x:x + w]

        # Verificar tamaño mínimo para evitar errores en embeddings
        if rostro.size < 50 * 50:
            print(f"Rostro {count} en {img_file} demasiado pequeño para obtener embeddings.")
            continue

        try:
            # Generar embeddings usando DeepFace
            embeddings = DeepFace.represent(img_path=rostro, model_name='Facenet512', enforce_detection=False)[0]['embedding']
            embedding_array = np.array(embeddings).reshape(1, -1)

            # Clasificar el rostro
            prediction = classifier.predict(embedding_array)
            predicted_label = label_encoder.inverse_transform(prediction)[0]

            # Almacenar similitudes
            if predicted_label not in detecciones:
                detecciones[predicted_label] = []
            detecciones[predicted_label].append((embeddings, (x, y, w, h), rostro))

        except Exception as e:
            print(f"Rostro en {img_file} no reconocido. Error: {str(e)}")

    return detecciones

def draw_detections(frame, detecciones, faces):
    seleccionados = {}
    rostros_no_reconocidos = []  # Lista para rostros no reconocidos

    for label, detecciones_rostros in detecciones.items():
        # Encontrar el rostro con mayor similitud (esto puede variar dependiendo de cómo se defina la similitud)
        mejor_rostro = max(detecciones_rostros, key=lambda d: d[0])  # Aquí se asume que mayor valor es mejor
        seleccionados[label] = mejor_rostro

    # Procesar rostros no reconocidos
    for face_data in faces:
        facial_area = face_data['facial_area']
        x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']

        # Verificar si el rostro ya está en seleccionados
        if not any((x, y, w, h) == (selected[1][0], selected[1][1], selected[1][2], selected[1][3]) for selected in seleccionados.values()):
            rostros_no_reconocidos.append((x, y, w, h))

    # Dibujar rectángulos y etiquetas
    for label, (embeddings, (x, y, w, h), rostro) in seleccionados.items():
        # Dibujar rectángulo verde para el rostro seleccionado
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Dibujar rectángulos rojos para los rostros no reconocidos
    for (x, y, w, h) in rostros_no_reconocidos:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Rectángulo rojo

def save_result_image(frame, img_file):
    result_image_path = os.path.join(result_path, f'procesado_{img_file}')
    cv2.imwrite(result_image_path, frame)
    print(f"Imagen procesada guardada en: {result_image_path}")

# Procesar cada imagen en la carpeta de pruebas
for img_file in image_files:
    print(f"\nProcesando imagen: {img_file}")
    image_path = os.path.join(test_path, img_file)

    frame = process_image(image_path)
    if frame is None:
        continue

    faces = detect_faces(frame, img_file)
    if not faces:
        print(f"No se detectaron rostros en la imagen: {img_file}")
        continue

    detecciones = process_faces(frame, faces, img_file)
    draw_detections(frame, detecciones, faces)
    save_result_image(frame, img_file)

print("Proceso de detección y almacenamiento de rostros completado.")
