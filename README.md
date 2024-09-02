# IA-de-reconocimiento
Proyecto de reconocimiento de estudiantes para asistencia autónoma.

## Flujo de Trabajo
![Diagrama de Flujo](ImagenesPrueba/flujo.png)

---

## Configuración en Windows

### 1. Instalar Python
Primero, asegúrate de instalar la última versión de Python. En este proyecto, se utilizó Python 3.12.

### 2. Configurar el Entorno de Desarrollo

#### 2.1. Ubicación del Proyecto
Ubícate en la carpeta donde se encuentra almacenado el proyecto:

```bash
cd D:\2024\Reconocimiento
```

#### 2.2. Crear un Entorno Virtual
Crea un entorno virtual para las pruebas con Python:

```bash
python -m venv myenv
```

#### 2.3. Brindar Permisos de Ejecución (si es necesario)
Es posible que necesites ajustar las políticas de ejecución para activar el entorno:

```bash
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
```

#### 2.4. Activar el Entorno Virtual
Activa el entorno recién creado:

```bash
myenv\Scripts\activate
```

---

## Pruebas de Librerías

### 3. Prueba con OpenCV y MediaPipe

En este apartado, se realizaron pruebas con las versiones y librerías necesarias para diferentes configuraciones de FaceID.

#### 3.1. Configuración de Python
Asegúrate de tener la versión correcta de Python:

```bash
python=3.12.3
```

#### 3.2. Instalación de Librerías

**Para FaceID:**
```bash
python -m pip install --upgrade pip
pip install --upgrade opencv-python-headless
```

**Para FaceID2:**
```bash
pip install opencv-python
pip install --upgrade opencv-contrib-python
pip install opencv-python mediapipe
```

**Para FaceID4:**
```bash
pip install opencv-python opencv-python-headless numpy
```

### 4. Prueba con DSFD 

Esta sección cubre la instalación y configuración de DSFD  para pruebas.

#### 4.1. Requisitos de Python
DSFD  requiere una versión específica de Python:

```bash
python 3.8
```

#### 4.2. Instalación de Librerías Necesarias
Instala las librerías requeridas:

```bash
# Pytorch, torchvision, torchaudio y cudatoolkit
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch  

# OpenCV
conda install conda-forge::opencv
```

#### 4.3. Instalar el Repositorio de DSFD
Clona e instala el repositorio:

```bash
pip install git+https://github.com/hukkelas/DSFD-Pytorch-Inference.git
```

#### 4.4. Ejecución de Pruebas
Para correr el programa:

- Asegúrate de tener el archivo `test.py` y un directorio llamado `images/` con las imágenes a procesar.
- Las imágenes procesadas se almacenarán en el mismo directorio.

Ejecuta el script con:

```bash
python3 test.py
```

### 5. Prueba en Jupyter con RetinaFace

Esta prueba se realiza dentro de un entorno específico con Jupyter y RetinaFace.

#### 5.1. Creación del Entorno
Crea un entorno de Conda:

```bash
conda create --name face python=3.10.8
```

#### 5.2. Instalación de Librerías
Instala las librerías necesarias:

```bash
python -m pip install -U pip
pip install "numpy<2.0"
pip install sympy
pip install pillow
pip install matplotlib
pip install opencv-python
python -m pip install -U matplotlib
pip install mtcnn
conda install -c conda-forge notebook
pip install retina-face
```

#### 5.3. Instalación de TensorFlow
Instala TensorFlow con soporte CUDA:

```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
python -m pip install "tensorflow<2.11"
```

---

## Ejecución de Código

Para ejecutar cualquier prueba de código en el entorno activo:

```bash
(myenv)[ruta actual del proyecto]> python [nombre del ejecutable].py
```

> **NOTA:**
> Cada prueba de librería se realizó en un entorno `myenv` diferente para controlar las versiones y asegurar que solo se instalaran las librerías necesarias.
