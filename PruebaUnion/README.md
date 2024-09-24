# Proyecto de Detección y Entrenamiento de Rostros

Este proyecto se enfoca en la detección de rostros a partir de imágenes y el entrenamiento de un modelo de reconocimiento facial. Utiliza la librería OpenCV para la manipulación de imágenes y la detección de rostros, con un enfoque específico en la recolección de datos y el preprocesamiento de imágenes. Tambien se busca combinar el uso de las librerias ya definidas para la deteccion de rostros.

## Archivos Incluidos

### 1. `Base_Data_Image.py`
Este script se encarga de:
- Detectar rostros en una carpeta de imágenes proporcionada.
- Redimensionar las imágenes varias veces hasta que se detecte al menos un rostro o se alcance un número máximo de intentos.
- Guardar las imágenes de los rostros detectados en una carpeta específica para su posterior uso en entrenamiento.

#### Proceso:
- Se especifica el nombre de la persona (`personName`) y las rutas de las imágenes y la carpeta de almacenamiento de rostros.
- Si no se detecta un rostro en la imagen, el script intenta redimensionarla varias veces (hasta 5 intentos) incrementando su tamaño en cada intento.
- Cuando se detecta un rostro, se recorta y redimensiona a 720x720 píxeles y se guarda en la carpeta de destino.

### 2. `Entrenamiento.py`
Este script se encarga de:
- Leer las imágenes de rostros previamente almacenadas en diferentes carpetas (una por cada persona).
- Preparar los datos (imágenes y etiquetas) para el entrenamiento de un modelo de reconocimiento facial.

#### Proceso:
- Se especifica la ruta donde se encuentran las carpetas con los rostros de diferentes personas (`dataPath`).
- Se recorren todas las imágenes dentro de las carpetas de personas, y se almacenan en una lista las imágenes junto con sus etiquetas correspondientes.
- Estas imágenes se usarán para entrenar un modelo de reconocimiento facial en pasos futuros.


markdown
Copiar código

## Dependencias

Este proyecto requiere las siguientes dependencias:
- pip install imutils
- pip install face-recognition
- pip install --upgrade opencv-python
- pip install opencv-contrib-python

Para instalarlas, puedes usar pip:
```bash
pip install opencv-python imutils
```
Uso
1. Detección y almacenamiento de rostros
Ejecuta el script Base_Data_Image.py para detectar rostros en las imágenes de una carpeta y almacenarlos en la carpeta correspondiente para cada persona.

```bash
python Base_Data_Image.py
```
2. Preparación para el entrenamiento
Ejecuta el script Entrenamiento.py para leer las imágenes de los rostros y preparar los datos para el entrenamiento, el entrenamiento de se hiso para identificar a 3 estudiante, con alrededor de 30 imagenes por estudiante, una muestra pequeña pero es para realizar pruebas.

```bash
python Entrenamiento.py
```

##### Comparacion de Modelos de Entrenamiento
###### 1. EigenFaceRecognizer
El modelo EigenFaceRecognizer se basa en la técnica de descomposición de imágenes mediante el análisis de componentes principales (PCA). Aquí, las imágenes de rostros se representan como combinaciones lineales de "rostros básicos" (eigenfaces), que son los componentes principales de la variabilidad de los rostros en el conjunto de datos de entrenamiento.
###### 2. FisherFaceRecognizer
Este modelo utiliza Análisis Discriminante Lineal (LDA) para mejorar la discriminación entre clases de rostros. Mientras que el EigenFace se basa en PCA (que solo reduce la dimensionalidad), FisherFace también maximiza la separación entre las clases (personas) en el espacio de características. Este método busca proyectar los datos a un espacio de menor dimensión donde las clases (personas) se separan lo máximo posible.

###### 3. LBPHFaceRecognizer (Local Binary Patterns Histogram)
El modelo LBPH se basa en características locales en lugar de análisis global como PCA o LDA. Utiliza patrones binarios locales (LBP), que se centran en la textura de las imágenes. Los píxeles en la vecindad de cada píxel central se comparan para formar un patrón binario, que luego se convierte en un histograma que describe la textura de la imagen.

###### Comparativa y Mejor Opción
- EigenFaceRecognizer es el más rápido, pero sensible a variaciones de condiciones, lo que lo hace adecuado solo para entornos controlados.
- FisherFaceRecognizer es más robusto que EigenFace, ya que maneja mejor las variaciones en iluminación y expresión. Se recomienda en situaciones donde haya más de una clase (varias personas) y las condiciones varíen ligeramente.
-LBPHFaceRecognizer es el más robusto frente a variaciones de iluminación, postura y expresiones, lo que lo convierte en la opción más versátil y precisa en situaciones reales con condiciones no controladas.
###### Conclusión:
Escogimos LBPHFaceRecognizer ya que suele ser la mejor opción para la mayoría de las aplicaciones prácticas, ya que maneja variaciones más comunes en los rostros del mundo real (como cambios de iluminación y expresiones faciales). Aunque es más lento y puede consumir más almacenamiento, su precisión y flexibilidad lo hacen ideal para escenarios no controlados.

### 2. `Reconocimiento_Facial.py`
En esta parte se implementa el modelo creado de `Entrenamiento.py` llamado `ModelFaceFrontalData2024.xml`, para el reconocimieto de los estudiantes, se ajusta el valor de confianza en un rango que puede variar entre 80-85, hay que tener en cuenta que este codigo es para prueba por lo tanto no se implemnetado la union las librerias de reconocimiento de rostros.