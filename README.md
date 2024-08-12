# IA-de-reconocimiento
Proyecto de reconocimiento de estudiantes para asistencia autónoma
# Flujo de Trabajo
![Diagrama de Flujo](ImagenesPrueba\Untitled-2024-08-11-1437.png)
# Windows

# Instalar python, si estas en window intalar 
     la ultima version de python en nuestro caso la version de python 3.12

# En la terminal Ubicarse en la carpeta donde se tenga almacenado el proyecto
    cd D:\2024\Reconocimiento

# Crear un entorno para realizar las pruebas con python
    python -m venv myenv

# Si es necesario brindar los permisos de politica para activar el entorno
    Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process

# Para activar el Scripts
    myenv\Scripts\activate

# Prueba de libreria de opencv y mediapipe
    #Instalar las librerias necesarias

    #Para FaceID
    python -m pip install --upgrade pip
    pip install --upgrade opencv-python-headless

    #Para FaceID2
    pip install opencv-python
    pip install --upgrade opencv-contrib-python
    pip install opencv-python mediapipe

    #Para FaceID4
    pip install opencv-python opencv-python-headless numpy

# Prueba de libreria DeepFace
    pip install deepface
    pip uninstall tensorflow
    pip install tensorflow
    pip install --upgrade tensorflow
    pip install --upgrade deepface
# Si se quiere ejecutar el codigo para alguna prueba 
    (myenv)[ruta actual del proyecto]> python [nombre del ejecutable].py