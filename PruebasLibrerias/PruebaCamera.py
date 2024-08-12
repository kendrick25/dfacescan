import cv2 as opencv

cap = opencv.VideoCapture(0)

if not cap.isOpened():
    print("No se puede abrir la cámara.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se puede leer el frame de la cámara.")
        break

    opencv.imshow("Camera", frame)

    if opencv.waitKey(1) & 0xFF == 27:  # Presiona Esc para salir
        break

cap.release()
opencv.destroyAllWindows()