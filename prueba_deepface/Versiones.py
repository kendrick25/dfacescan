import os
import sys
import tensorflow as tf
from deepface import DeepFace
import numpy as np
import cv2
from PIL import Image, ImageOps
import matplotlib
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Deshabilita el uso de GPU
tf.get_logger().setLevel('ERROR')  # Suprime advertencias de TensorFlow

warnings.filterwarnings("ignore")  # Suprime advertencias generales

# Mostrar versiones de las librerías instaladas
print(f"Versión de Python: {sys.version}")
print(f"Versión de NumPy: {np.__version__}")
print(f"Versión de OpenCV: {cv2.__version__}")
print(f"Versión de Pillow: {Image.__version__}")
print(f"Versión de Matplotlib: {matplotlib.__version__}")
print(f"Versión de Deepface: {DeepFace.__version__}")

# Imprimir la versión de TensorFlow
print("Versión de TensorFlow:", tf.__version__)

# Comprobar la lista de dispositivos disponibles
print("Dispositivos disponibles:", tf.config.list_physical_devices())
