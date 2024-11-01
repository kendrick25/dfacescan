import os  
import cv2
import joblib
import numpy as np
from deepface import DeepFace

# Cargar el clasificador entrenado y el codificador de etiquetas
print("Cargando el clasificador y el codificador de etiquetas...")
clf = joblib.load('./models/face_classifier.pkl')
le = joblib.load('./models/label_encoder.pkl')

# Definir rutas
data_path = "./data"  
test_path = "./test"  

# Crear carpeta 'result' si no existe
if not os.path.exists('./result'):
    os.makedirs('./result')
print("Carpeta 'result' creada (si no existía).")

# Obtener todas las imágenes en la carpeta test
image_files = [f for f in os.listdir(test_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Verificar que haya imágenes para procesar
if not image_files:
    print("No hay imágenes en la carpeta de prueba.")
    exit()
else:
    print(f"Se encontraron {len(image_files)} imágenes para procesar.")

# Procesar cada imagen en la carpeta de pruebas
for img_file in image_files:
    print(f"\nProcesando imagen: {img_file}")
    image_path = os.path.join(test_path, img_file)
    image = cv2.imread(image_path)

    # Verificar si la imagen se cargó correctamente
    if image is None:
        print(f"Error al cargar la imagen {img_file}.")
        continue

    # Detectar rostros en la imagen usando RetinaFace
    print("Detectando rostros en la imagen...")
    faces = DeepFace.extract_faces(img_path=image_path, detector_backend='retinaface', enforce_detection=True)
    print(f"{len(faces)} rostros detectados.")

    # Listas para almacenar probabilidades y IDs de usuarios
    valid_probabilities = []
    valid_user_ids = []

    # Procesar cada rostro detectado
    for idx, face_data in enumerate(faces):
        print(f"\nProcesando rostro {idx + 1}/{len(faces)}:")
        facial_area = face_data['facial_area']
        face = face_data['face']

        # Redimensionar el rostro extraído para el modelo
        face_resized = cv2.resize(face, (160, 160))
        print("Rostro redimensionado para el modelo.")

        # Generar el embedding usando DeepFace
        try:
            embedding = DeepFace.represent(face_resized, model_name="Facenet512", enforce_detection=False)[0]["embedding"]
            embedding = np.array(embedding).reshape(1, -1)  # Asegura que sea de forma (1, 512)
            print("Embedding generado.")
        except Exception as e:
            print(f"Error al generar el embedding para {img_file}: {e}")
            continue

        # Realizar predicción
        probabilities = clf.predict_proba(embedding)
        min_prob_idx = np.argmin(probabilities)
        min_prob = probabilities[0, min_prob_idx]
        print(f"Probabilidad más baja de predicción: {min_prob:.2f} (Índice: {min_prob_idx})")

        # Aplicar umbral dinámico
        threshold = 0.1
        if min_prob > threshold:
            valid_probabilities.append(min_prob)
            user_id = le.inverse_transform([min_prob_idx])[0]  # Obtener el ID del usuario
            valid_user_ids.append(user_id)
            print(f"Rostro {idx + 1} cumple con el umbral con ID: {user_id} y probabilidad: {min_prob:.2f}")
        else:
            print(f"Rostro {idx + 1} no cumple con el umbral ({threshold}).")

    # Determinar el rostro correcto basado en la probabilidad mínima
    if valid_probabilities:
        # Encontrar el índice de la probabilidad mínima por encima del umbral
        best_index = np.argmin(valid_probabilities)
        best_prob = valid_probabilities[best_index]
        best_user_id = valid_user_ids[best_index]

        label = f"ID: {best_user_id} ({best_prob:.2f})"
        print(f"Mejor rostro seleccionado - ID: {best_user_id}, Probabilidad: {best_prob:.2f}")
        
        # Usar el área facial del rostro seleccionado
        best_facial_area = faces[best_index]['facial_area']
        x, y, w, h = best_facial_area['x'], best_facial_area['y'], best_facial_area['w'], best_facial_area['h']
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1)
    else:
        label = "No match"
        print("Ningún rostro cumple con el umbral. Etiquetado como 'No match'.")
        cv2.putText(image, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Guardar la imagen con resultados en la carpeta 'result'
    output_image_path = os.path.join('./result', f'result_{img_file}')
    cv2.imwrite(output_image_path, image)
    print(f"Resultado guardado en: {output_image_path}")
