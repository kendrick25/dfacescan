import os
import cv2

def load_data(data_dir):
    print("Iniciando carga de datos...")
    images = []
    labels = []
    
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            print(f"Cargando imágenes de: {label}")
            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img_path)  # Guarda la ruta de la imagen
                    labels.append(label)      # Guarda la etiqueta
                else:
                    print(f"Error: No se pudo cargar la imagen {img_path}")

    print(f"Cargadas {len(images)} imágenes de {len(set(labels))} personas.")  # Mensaje de carga
    return images, labels
