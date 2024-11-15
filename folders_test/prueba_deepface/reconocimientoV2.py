import os 
import tensorflow as tf
import warnings
import cv2
import numpy as np
from deepface import DeepFace

# Deshabilita el uso de GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  
# Suprime advertencias de TensorFlow
tf.get_logger().setLevel('ERROR')  
# Suprime advertencias generales
warnings.filterwarnings("ignore")  

# Ruta de la carpeta que contiene las imágenes de prueba
test_folder = "D:/Nueva carpeta/Archivos UTP/PruebaDeepface/test/"
# Ruta de la carpeta que contiene los rostros objetivo (subcarpetas de diferentes personas)
data_folder = "D:/Nueva carpeta/Archivos UTP/PruebaDeepface/data/"
output_folder = "D:/Nueva carpeta/Archivos UTP/PruebaDeepface/Resultados/"  # Ruta para guardar las imágenes resultantes

def extract_faces(image_path):
    """Extrae rostros de la imagen dada."""
    print(f"Extracting faces from the original image: {image_path}")
    faces = DeepFace.extract_faces(img_path=image_path, detector_backend='retinaface', enforce_detection=True)
    return faces

def compare_faces(faces, target_face):
    """Compara los rostros detectados con el rostro objetivo y devuelve el mejor resultado."""
    best_result = None  
    best_distance = float('inf')  
    best_index = -1  

    for i, face_data in enumerate(faces):
        facial_area = face_data['facial_area']  
        face = face_data['face']  

        # Obtener la información del rostro detectado
        R, G, B = face.T
        __bgr_face = np.array((B, G, R)).T
        bgr_face = (__bgr_face * 255).astype(np.uint8)

        # Realizar la comparación
        result = DeepFace.verify(
            img1_path=target_face,
            img2_path=bgr_face,
            detector_backend="skip",
            model_name="Facenet512",
            distance_metric="cosine",
            threshold=0.4,  # Ajustar el umbral
            enforce_detection=True
        )

        # Verificar si este es el mejor resultado
        if result['distance'] < best_distance:
            best_distance = result['distance']
            best_result = result
            best_index = i

        print(f"Comparing with target face: Verified: {result['verified']}, Distance: {result['distance']}")

    return best_result, best_index

def process_images():
    """Proceso principal para extraer y comparar rostros en las imágenes."""
    # Inicializar la lista para almacenar los resultados
    results = []

    # Iterar a través de cada archivo en la carpeta de test
    for filename in os.listdir(test_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Solo procesar imágenes
            image_path = os.path.join(test_folder, filename)
            image = cv2.imread(image_path)

            # Extraer los rostros de la imagen original
            faces = extract_faces(image_path)

            # Iterar a través de cada subcarpeta en la carpeta data
            for person_folder in os.listdir(data_folder):
                person_folder_path = os.path.join(data_folder, person_folder)

                if os.path.isdir(person_folder_path):  # Verificar que sea un directorio
                    # Solo tomar la primera imagen de cada persona
                    for data_filename in os.listdir(person_folder_path):
                        if data_filename.endswith(".jpg") or data_filename.endswith(".png"):  # Solo procesar imágenes
                            face_target_path = os.path.join(person_folder_path, data_filename)

                            # Extraer el rostro objetivo (face_target) de la imagen de referencia
                            print(f"Extracting target face from: {face_target_path}")
                            target_face_data = DeepFace.extract_faces(img_path=face_target_path, detector_backend='retinaface', enforce_detection=True)

                            # Asegurarse de que se detectó el rostro objetivo
                            if len(target_face_data) > 0:
                                target_face = target_face_data[0]['face']  # Obtener el rostro objetivo
                                
                                # Comparar los rostros
                                best_result, best_index = compare_faces(faces, target_face)

                                # Guardar resultados solo si se encontró un mejor resultado
                                if best_result is not None:
                                    facial_area = faces[best_index]['facial_area']
                                    results.append([person_folder, data_filename, filename, best_index, best_result['distance'], best_result['threshold'], 'cosine'])
                                    
                                    # Dibujar un rectángulo verde si coinciden, rojo si no coinciden
                                    x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                                    color = (0, 255, 0) if best_result['verified'] else (0, 0, 255)

                                    # Dibujar el rectángulo en la imagen original
                                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                                
                                # Solo extraer un rostro por persona
                                break

            # Guardar la imagen resultante en la ruta especificada
            output_path = os.path.join(output_folder, f"result_{filename}")
            cv2.imwrite(output_path, image)
            print(f"Image saved at: {output_path}")

    print("\nProcessing complete.")

# Ejecutar el procesamiento de imágenes
process_images()
