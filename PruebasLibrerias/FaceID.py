import cv2 as opencv

# Cargar el clasificador en cascada preentrenado para la detección de rostros
face_cascade = opencv.CascadeClassifier(opencv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Iniciamos la captura de video
cap = opencv.VideoCapture(0)

# Configura la resolución de la captura de video
desired_width = 640
desired_height = 480
cap.set(opencv.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(opencv.CAP_PROP_FRAME_HEIGHT, desired_height)

# Crear un diccionario para rastreadores y números de identificación
trackers = {}
tracker_id = 0
frame_count = 0
detect_interval = 30  # Intervalo de frames para la detección de rostros

# Función para dibujar texto centrado
def draw_label(frame, text, x, y):
    font = opencv.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    color = (0, 255, 0)
    thickness = 1
    size = opencv.getTextSize(text, font, scale, thickness)[0]
    x_center = x + (size[0] // 2)
    y_center = y - 10
    opencv.putText(frame, text, (x_center, y_center), font, scale, color, thickness, opencv.LINE_AA)

# Proceso principal
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Detectar rostros cada cierto número de frames
    if frame_count % detect_interval == 0:
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Añadir nuevos rastreadores para rostros detectados
        for (x, y, w, h) in faces:
            # Verificar si el rostro ya está siendo rastreado
            matched = False
            for fid, tracker in trackers.items():
                success, bbox = tracker.update(frame)
                if success:
                    tx, ty, tw, th = [int(v) for v in bbox]
                    if abs(x - tx) < w and abs(y - ty) < h:
                        matched = True
                        break
            if not matched:
                tracker = opencv.TrackerKCF_create()
                tracker.init(frame, (x, y, w, h))
                trackers[tracker_id] = tracker
                tracker_id += 1

    # Actualizar rastreadores
    new_trackers = {}
    for fid, tracker in trackers.items():
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            new_trackers[fid] = tracker
            draw_label(frame, f"ID: {fid}", x, y)
            opencv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Reemplazar rastreadores antiguos con los actualizados
    trackers = new_trackers

    # Mostrar el frame
    opencv.imshow("Detect Camera", frame)

    # Salida con la tecla Esc
    if opencv.waitKey(5) == 27:
        break

# Liberar recursos
opencv.destroyAllWindows()
cap.release()