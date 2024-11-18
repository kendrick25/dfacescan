import os
import cv2
import joblib
import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
import sys

# Rutas de entrada y salida
test_path = "./test"
result_path = "result"
model_path = './models/face_classifier.pkl'
encoder_path = './models/label_encoder.pkl'

# Crear carpeta para almacenar la imagen resultante si no existe
os.makedirs(result_path, exist_ok=True)

# Cargar el clasificador y el codificador de etiquetas
classifier = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)

# Lista de detectores para probar
detectors = ['retinaface', 'mtcnn', 'opencv', 'ssd', 'dlib']

# Definir un tamaño mínimo para los rostros
MIN_FACE_WIDTH = 30
MIN_FACE_HEIGHT = 30

def detect_faces(img_file):
    """Detecta rostros en la imagen proporcionada usando DeepFace."""
    try:
        # Cargar la imagen
        img_path = os.path.join(test_path, img_file)
        img = cv2.imread(img_path)

        # Detectar rostros usando DeepFace con extract_faces
        faces = DeepFace.extract_faces(img_path, detector_backend='retinaface')
        
        # Si no se detectan rostros, faces será una lista vacía
        if not faces:
            print("[INFO] No se detectaron rostros en la imagen.")
            return img, []

        # Devolver la imagen y las caras detectadas
        print(f"[INFO] Rostros detectados: {len(faces)}")
        return img, faces
    except Exception as e:
        print(f"[ERROR] Error al detectar rostros: {e}")
        return None, []
    
def generate_embeddings(frame, faces):
    """Genera los embeddings de cada rostro detectado, creando embeddings vacíos para rostros que no cumplen con el tamaño mínimo."""
    embeddings = []
    total = len(faces)
    
    for count, face_data in enumerate(faces):
        sys.stdout.write(f"\rGenerando embeddings para rostro {count + 1}/{total}...")
        sys.stdout.flush()

        try:
            # Extraer área facial y coordenadas
            facial_area = face_data['facial_area']
            if not isinstance(facial_area, dict):
                continue  # Si no es un diccionario, omitir este rostro
            
            x, y, w, h = facial_area.get('x', 0), facial_area.get('y', 0), facial_area.get('w', 0), facial_area.get('h', 0)

            # Verificar si el tamaño cumple con los requisitos mínimos
            if w < MIN_FACE_WIDTH or h < MIN_FACE_HEIGHT:
                # Crear un embedding "vacío" (vector de ceros) para rostros que no cumplen con el tamaño mínimo
                empty_embedding = np.zeros(512)  # Cambia 512 según el tamaño del modelo (Facenet512)
                embeddings.append((empty_embedding, (x, y, w, h)))
                continue

            # Extraer el rostro y generar embedding
            rostro = frame[y:y + h, x:x + w]
            face_embedding = DeepFace.represent(
                img_path=rostro, 
                model_name='Facenet512',
                enforce_detection=False,
                detector_backend='retinaface'
            )[0]['embedding']

            embeddings.append((face_embedding, (x, y, w, h)))  # Guardar embedding con coordenadas

        except Exception as e:
            # En caso de error, generar un embedding vacío para mantener la consistencia
            print(f"\n[WARN] Error al generar embedding para rostro {count + 1}: {e}")
            error_embedding = np.zeros(512)  # Cambia 512 según el tamaño del modelo (Facenet512)
            embeddings.append((error_embedding, (x, y, w, h)))
            
    print()  # Salto de línea después de mostrar la barra de progreso
    return embeddings

def heuristic(embeddings, known_embeddings, known_labels, classifier, label_encoder):
    results = []
    total = len(embeddings)

    # Umbrales ajustados
    SIMILARITY_THRESHOLD = 0.92   # Aumentamos el umbral de similitud 0.92
    DISTANCE_THRESHOLD = 20       # Reducimos el umbral de proximidad (más estricto) 20  
    CONFIDENCE_THRESHOLD = 0.75 # Nuevo umbral para la confianza del clasificador 0.75 

    assigned_labels = set()

    for count, (embedding, (x, y, w, h)) in enumerate(embeddings):
        sys.stdout.write(f"\rEvaluando heurística para rostro {count + 1}/{total}...")
        sys.stdout.flush()

        best_similarity = 0
        best_label = None
        match_index = None

        if len(known_embeddings) > 0:
            similarities = cosine_similarity([embedding], known_embeddings)[0]

            # Filtrar por similitud y distancia
            for idx, similarity in enumerate(similarities):
                known_position = known_labels[idx][1]
                distance = np.linalg.norm(np.array([x, y]) - np.array(known_position))

                if similarity > SIMILARITY_THRESHOLD and distance < DISTANCE_THRESHOLD:
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_label = known_labels[idx][0]
                        match_index = idx

        # Verificar coincidencia con el rostro conocido
        if match_index is not None and best_label not in assigned_labels:
            results.append({'face_area': (x, y, w, h), 'label': best_label})
            assigned_labels.add(best_label)
        else:
            # Clasificador como respaldo
            prediction = classifier.predict([embedding])
            confidence = np.max(classifier.predict_proba([embedding]))  # Confianza del clasificador
            predicted_label = label_encoder.inverse_transform(prediction)[0]

            if confidence >= CONFIDENCE_THRESHOLD:
                # Si el clasificador tiene suficiente confianza
                known_embeddings.append(embedding)
                known_labels.append((predicted_label, (x, y)))
                results.append({'face_area': (x, y, w, h), 'label': predicted_label})
            else:
                # Marca el rostro como desconocido si no hay confianza suficiente
                results.append({'face_area': (x, y, w, h), 'label': "desconocido"})

    print()
    return results

def drawing(frame, results):
    """Dibuja los resultados sobre la imagen, diferenciando entre rostros conocidos y desconocidos."""
    for result in results:
        x, y, w, h = result['face_area']
        label = result['label']

        if label.lower() == "desconocido":
            label= ""
            # Color para rostros desconocidos (Rojo)
            color = (0, 0, 255)  # BGR: rojo
        else:
            # Color para rostros conocidos (Verde)
            color = (0, 255, 0)  # BGR: verde

        # Dibujar rectángulo alrededor del rostro
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Dibujar etiqueta sobre el rostro
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
def main():
    known_embeddings = []  # Inicializamos las listas fuera del bucle para acumular embeddings
    known_labels = []

    for img_file in os.listdir(test_path):
        if img_file.endswith(".jpg") or img_file.endswith(".png")or img_file.endswith(".jpeg"):
            print(f"[INFO] Procesando imagen: {img_file}")

            # Detectar rostros
            img, faces = detect_faces(img_file)

            # Generar embeddings
            embeddings = generate_embeddings(img, faces)

            # Conocer los resultados mediante la heurística
            results = heuristic(embeddings, known_embeddings, known_labels, classifier, label_encoder)

            # Verificar si 'results' tiene datos
            if not results:
                print("[INFO] No se encontraron rostros procesados.")
            else:
                # Imprimir los resultados de los rostros antes de dibujar
                print("\n[INFO] Resultados por rostro detectado:")
                for idx, result in enumerate(results, 1):
                    x, y, w, h = result['face_area']
                    label = result['label']
                    print(f"Rostro {idx}: Etiqueta: {label}, Coordenadas: ({x}, {y}, {w}, {h})")

            # Llamada a la función de dibujo
            drawing(img, results)

            result_img_path = os.path.join(result_path, f"result_{img_file}")
            cv2.imwrite(result_img_path, img)
            print(f"\n[INFO] Imagen procesada guardada en: {result_img_path}")

if __name__ == "_main_":
    main()