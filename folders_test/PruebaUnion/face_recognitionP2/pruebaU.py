import cv2
import face_recognition

# Imagen a comparar
######################################################################
image = cv2.imread("ImagenesPrueba/TAHIS/CERCA/11.jpg")  # Verifica la ruta de la imagen

# Detectar ubicaciones de rostros
face_locs = face_recognition.face_locations(image)

# Asegúrate de que se detectó al menos un rostro
if len(face_locs) > 0:
    # Obtén las codificaciones dels primer rostro
    face_image_encodings = face_recognition.face_encodings(image, known_face_locations=face_locs)[0]
else:
    raise ValueError("No se detectaron rostros en la imagen de referencia.")
#
# Video Streaming
######################################################################
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    # Detección de rostros en el cuadro del video
    face_locations = face_recognition.face_locations(frame)
    if face_locations:
        for face_location in face_locations:
            # Comparación
            face_frame_encodings = face_recognition.face_encodings(frame, known_face_locations=[face_location])[0]
            result = face_recognition.compare_faces([face_image_encodings], face_frame_encodings)
            print("Resultado:", result)
            if result[0]==True:
                text="Thais"
                Color =(125,228,0)
            else:
                text = "Desconocido"
                color = (50,50,255)

            # Dibujar un rectángulo alrededor del rostro
            cv2.rectangle(frame, (face_location[3],face_location[2]), (face_location[1],face_location[2]+30),color,-1)
            cv2.rectangle(frame, (face_location[3],face_location[0]), (face_location[1],face_location[2]),color,2)
            cv2.putText(frame, text, (face_location[3],face_location[2]+20),2,0.7,(255,255,255),1)
    # Mostrar el cuadro con los rostros detectados
    cv2.imshow("Frame", frame)

    # Salir al presionar la tecla ESC
    k = cv2.waitKey(1)
    if k == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
