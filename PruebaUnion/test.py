from retinaface import RetinaFace
from facenet_pytorch import InceptionResnetV1
import torch
import cv2
import numpy as np

# Inicializar FaceNet para extraer embeddings faciales
model = InceptionResnetV1(pretrained='vggface2').eval()

# Inicializar la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al acceder a la cámara")
        break

    # Detectar rostros con RetinaFace
    detections = RetinaFace.detect_faces(frame)

    # Procesar cada rostro detectado
    if isinstance(detections, dict):
        for key in detections.keys():
            identity = detections[key]
            facial_area = identity["facial_area"]  # Coordenadas del área del rostro

            # Extraer la región del rostro y redimensionar a 160x160 para FaceNet
            face = frame[facial_area[1]:facial_area[3], facial_area[0]:facial_area[2]]
            face = cv2.resize(face, (160, 160))
            face = torch.tensor(face).permute(2, 0, 1).unsqueeze(0).float() / 255.0

            # Obtener los embeddings faciales con FaceNet
            embeddings = model(face)

            # Dibujar el cuadro alrededor del rostro detectado
            cv2.rectangle(frame, (facial_area[0], facial_area[1]), (facial_area[2], facial_area[3]), (0, 255, 0), 2)

    # Mostrar la imagen con los rostros detectados
    cv2.imshow('Reconocimiento con RetinaFace y FaceNet', frame)

    # Presionar 'ESC' para salir
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
