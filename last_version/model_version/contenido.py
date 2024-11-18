import pickle
import os
from joblib import load

# Ruta a los archivos
face_classifier_path = "face_classifier.pkl"
label_encoder_path = "label_encoder.pkl"

def load_pickle_file(file_path):
    """Carga y devuelve el contenido de un archivo pickle."""
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except pickle.UnpicklingError:
        print(f"{file_path} no es un archivo pickle válido.")
    except FileNotFoundError:
        print(f"Archivo no encontrado: {file_path}")
    except Exception as e:
        print(f"Error al cargar {file_path}: {e}")
    return None

def load_with_joblib(file_path):
    """Carga un archivo utilizando joblib como método alternativo."""
    try:
        return load(file_path)
    except Exception as e:
        print(f"Error al cargar {file_path} con joblib: {e}")
    return None

def inspect_file(file_path):
    """Inspecciona los primeros bytes de un archivo para diagnosticar problemas."""
    try:
        with open(file_path, 'rb') as file:
            content = file.read(100)  # Leer los primeros 100 bytes
            print(f"Contenido inicial de {file_path}:\n{content}\n")
    except Exception as e:
        print(f"Error al inspeccionar {file_path}: {e}")

# Confirmar el directorio actual
print(f"Directorio actual: {os.getcwd()}")

# Inspeccionar los archivos para diagnosticar problemas
inspect_file(face_classifier_path)
inspect_file(label_encoder_path)

# Intentar cargar los archivos con pickle
print("Cargando archivos con pickle...")
face_classifier = load_pickle_file(face_classifier_path)
label_encoder = load_pickle_file(label_encoder_path)

# Si pickle falla, intentar cargar con joblib
if face_classifier is None:
    print("\nIntentando cargar face_classifier.pkl con joblib...")
    face_classifier = load_with_joblib(face_classifier_path)

if label_encoder is None:
    print("\nIntentando cargar label_encoder.pkl con joblib...")
    label_encoder = load_with_joblib(label_encoder_path)

# Mostrar información de face_classifier
print("\nContenido de face_classifier.pkl:")
if face_classifier:
    print(face_classifier)
    if hasattr(face_classifier, '__dict__'):
        print("\nAtributos de face_classifier:")
        print(face_classifier.__dict__)
else:
    print("No se pudo cargar face_classifier.pkl.")

# Mostrar información de label_encoder
print("\nContenido de label_encoder.pkl:")
if label_encoder:
    print(label_encoder)
    if hasattr(label_encoder, 'classes_'):
        print("\nClases en el label_encoder:")
        print(label_encoder.classes_)
else:
    print("No se pudo cargar label_encoder.pkl.")
