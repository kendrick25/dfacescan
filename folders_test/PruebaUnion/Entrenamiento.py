import cv2
import os
import numpy as np

dataPath = 'D:/Nueva carpeta/Archivos UTP/Robotica/Data'
peopleList = os.listdir(dataPath)
print('Lista de Personas: ', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir
    print('Leyendo Imagenes')

    for fileName in os.listdir(personPath):
        print('Rostros: ', nameDir + '/ ' + fileName)
        labels.append(label)

        facesData.append(cv2.imread(personPath + '/' + fileName, 0))
        image = cv2.imread(personPath + '/' + fileName, 0)
        # Verificar conteo de la base de datos de imágenes
        # cv2.imshow('image', image)
        # cv2.waitKey(10)
    label = label + 1

# cv2.destroyAllWindows()

# Crear el modelo LBPH
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

print("Entrenando...")
face_recognizer.train(facesData, np.array(labels))

# Guardar el modelo entrenado
face_recognizer.write('ModelFaceFrontalData2024.xml')
print("Modelo Guardado")
