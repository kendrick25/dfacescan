# IA-de-reconocimiento
Proyecto de reconocimiento de estudiantes para asistencia autónoma

# Flujo de Trabajo
![Diagrama de Flujo](ImagenesPrueba/flujo.png)

# Windows

# Instalar Python
Si estás en Windows, instala la última versión de Python, en nuestro caso la versión de Python 3.12.

# En la terminal, ubicarse en la carpeta donde se tenga almacenado el proyecto
    cd D:\2024\Reconocimiento

# Crear un entorno para realizar las pruebas con Python
    python -m venv myenv

# Si es necesario, brindar los permisos de política para activar el entorno
    Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process

# Para activar el entorno
    myenv\Scripts\activate

# Prueba de librería de OpenCV y MediaPipe
    # Instalar las librerías necesarias

    # Para FaceID
    python -m pip install --upgrade pip
    pip install --upgrade opencv-python-headless

    # Para FaceID2
    pip install opencv-python
    pip install --upgrade opencv-contrib-python
    pip install opencv-python mediapipe

    # Para FaceID4
    pip install opencv-python opencv-python-headless numpy

# Prueba de librería DeepFace
    pip install deepface
    pip uninstall tensorflow
    pip install tensorflow
    pip install --upgrade tensorflow
    pip install --upgrade deepface

# Si se quiere ejecutar el código para alguna prueba
    (myenv)[ruta actual del proyecto]> python [nombre del ejecutable].py