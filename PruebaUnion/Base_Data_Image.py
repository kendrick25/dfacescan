import cv2
import os
import imutils

###############################################################
personName = 'TAHIS'
dataPath = 'D:\\Nueva carpeta\\Archivos UTP\\Robotica\\Data' 
personPath = dataPath + '/' + personName

# Ruta de la carpeta de imágenes
imagesPath = 'D:\\Nueva carpeta\\Archivos UTP\\Robotica\\imgenesEstudiantes\\TAHIS'

# Crear la carpeta para guardar los rostros si no existe
if not os.path.exists(personPath):
    print('Carpeta Creada: ', personPath)
    os.makedirs(personPath)

################################################################
# Obtener la lista de todas las imágenes en la carpeta
images = [f for f in os.listdir(imagesPath) if os.path.isfile(os.path.join(imagesPath, f))]
count = 0
total_images = len(images)
max_resize_attempts = 5  # Máximo número de intentos de redimensionamiento

# Cargar el clasificador de rostros
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Procesar cada imagen en la carpeta
for image_name in images:
    # Leer la imagen
    image_path = os.path.join(imagesPath, image_name)
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"No se pudo leer la imagen {image_name}")
        continue

    resize_attempts = 0  # Contador de intentos de redimensionamiento
    faces_detected = False  # Bandera para indicar si se ha detectado un rostro

    # Bucle para intentar redimensionar la imagen varias veces hasta detectar un rostro
    while resize_attempts < max_resize_attempts and not faces_detected:
        # Redimensionar la imagen para estandarizar el tamaño
        frame_resized = imutils.resize(frame, width=320 + resize_attempts * 100)  # Aumenta el tamaño en cada intento
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
        auxFrame = frame_resized.copy()

        # Detectar rostros en la imagen
        faces = faceClassif.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:  # Si se detectan rostros
            faces_detected = True
            for (x, y, w, h) in faces:
                # Recortar y redimensionar el rostro detectado
                rostro = auxFrame[y:y + h, x:x + w]
                rostro = cv2.resize(rostro, (720, 720), interpolation=cv2.INTER_CUBIC)
                
                # Guardar el rostro detectado en la carpeta
                cv2.imwrite(os.path.join(personPath, f'rostro_{count}.jpg'), rostro)
                count += 1
        else:
            print(f"No se detectaron rostros en la imagen {image_name}, intento {resize_attempts+1}")

        # Incrementar el contador de intentos de redimensionamiento
        resize_attempts += 1

    if not faces_detected:
        print(f"No se detectaron rostros después de {max_resize_attempts} intentos en la imagen {image_name}.")

    # Condición de salida si se alcanzan todas las imágenes
    if count >= total_images:
        break

# No es necesario cerrar ventanas ya que no se están mostrando imágenes
