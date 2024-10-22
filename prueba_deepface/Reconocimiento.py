import os
import tensorflow as tf
import warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Deshabilita el uso de GPU
tf.get_logger().setLevel('ERROR')  # Suprime advertencias de TensorFlow
warnings.filterwarnings("ignore")  # Suprime advertencias generales
import cv2
import numpy as np
from deepface import DeepFace

# Métricas y modelos disponibles
metrics = ["cosine", "euclidean", "euclidean_l2"]
models = ["Facenet", "Facenet512", "VGG-Face", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace", "GhostFaceNet"]

# Ruta de la imagen original y del rostro objetivo (face_target)
image_path = "D:/Nueva carpeta/Archivos UTP/prueba_deepface/test/ik.jpeg"
face_target_path = "D:/Nueva carpeta/Archivos UTP/prueba_deepface/data/Thais/11.jpg"
output_path = "D:/Nueva carpeta/Archivos UTP/prueba_deepface/Resultados/result_img-00.jpg"  # Ruta para guardar la imagen resultante

# Cargar la imagen original
image = cv2.imread(image_path)

# Extraer los rostros de la imagen original
print("Extracting faces...")
faces = DeepFace.extract_faces(img_path=image_path, detector_backend='retinaface', enforce_detection=True)

# Extraer el rostro objetivo (face_target) de la imagen de referencia
target_face_data = DeepFace.extract_faces(img_path=face_target_path, detector_backend='retinaface', enforce_detection=True)

# Inicializar la lista para almacenar los resultados
results = []

# Asegurarse de que se detectó el rostro objetivo
if len(target_face_data) > 0:
    target_face = target_face_data[0]['face']  # Obtener el rostro objetivo

    R1, G1, B1 = target_face.T
    __bgr_target_face = np.array((B1, G1, R1)).T
    bgr_target_face = (__bgr_target_face * 255).astype(np.uint8)

    total_faces = len(faces)
    
    # Iterar sobre cada rostro detectado en la imagen
    for i, face_data in enumerate(faces):
        facial_area = face_data['facial_area']  # Obtener el área facial (bounding box)
        face = face_data['face']  # Obtener el rostro extraído

        R, G, B = face.T
        __bgr_face = np.array((B, G, R)).T
        bgr_face = (__bgr_face * 255).astype(np.uint8)

        # Realizar la comparación (verificación) entre el rostro detectado y el rostro objetivo
        result = DeepFace.verify(
            img1_path=bgr_target_face,
            img2_path=bgr_face,
            detector_backend="skip",  # Omitir la detección ya que están los rostros
            model_name="Facenet512",
            distance_metric="cosine",
            threshold=0.49,
            enforce_detection=True
        )

        results.append([i, result['distance'], result['threshold'], 'cosine'])

        # Dibujar un rectángulo verde si coinciden, rojo si no coinciden
        x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
        color = (0, 255, 0) if result['verified'] else (0, 0, 255)

        # Dibujar el rectángulo en la imagen original
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    # Guardar la imagen resultante en la ruta especificada
    cv2.imwrite(output_path, image)
    print(f"\nImage saved at: {output_path}")

else:
    print("The target face was not detected.")
