import cv2 as cv
import mediapipe as mp

# Inicializar MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Iniciar la captura de video y configurar resolución
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

# Crear un diccionario para rostros rastreados y números de identificación
face_ids = {}
next_face_id = 0
frame_count = 0
detect_interval = 30  # Intervalo de frames para la detección de rostros

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] + 1) * (boxA[3] + 1)
    boxBArea = (boxB[2] + 1) * (boxB[3] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def draw_label(frame, text, x, y):
    font = cv.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    color = (0, 255, 0)
    thickness = 1
    size = cv.getTextSize(text, font, scale, thickness)[0]
    x_center = x + (size[0] // 2)
    y_center = y - 10
    cv.putText(frame, text, (x_center, y_center), font, scale, color, thickness, cv.LINE_AA)

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Detectar rostros cada cierto número de frames
        if frame_count % detect_interval == 0:
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            new_faces = []
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = [int(v * dim) for v, dim in zip([bboxC.xmin, bboxC.ymin, bboxC.width, bboxC.height], [iw, ih, iw, ih])]
                    new_faces.append((x, y, w, h))

            # Asociar detecciones nuevas con rostros existentes
            updated_face_ids = {}
            for x, y, w, h in new_faces:
                cx, cy = x + w // 2, y + h // 2
                matched_id = None
                best_iou = 0

                for fid, (fx, fy, fw, fh) in face_ids.items():
                    fcx, fcy = fx + fw // 2, fy + fh // 2
                    iou = calculate_iou((x, y, w, h), (fx, fy, fw, fh))
                    if iou > best_iou:
                        best_iou = iou
                        matched_id = fid

                if best_iou > 0.5:
                    updated_face_ids[matched_id] = (x, y, w, h)
                else:
                    # Asignar un nuevo ID si no se encontró una coincidencia adecuada
                    matched_id = next_face_id
                    next_face_id += 1
                    updated_face_ids[matched_id] = (x, y, w, h)

            face_ids = updated_face_ids

        # Dibujar el rostro y el ID
        for fid, (x, y, w, h) in face_ids.items():
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            draw_label(frame, f"ID: {fid}", x, y)

        # Mostrar el frame y salir con la tecla Esc
        cv.imshow("Detect Camera", frame)
        if cv.waitKey(5) == 27:
            break

# Liberar recursos
cap.release()
cv.destroyAllWindows()