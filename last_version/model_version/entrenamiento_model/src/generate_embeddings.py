from deepface import DeepFace
import numpy as np
import os
import json
from datetime import datetime
import time  # Importar módulo de tiempo
from load_data import load_data

def generate_embeddings(images):
    print("Generando embeddings...")
    embeddings = []
    for img in images:
        # Usamos DeepFace.represent para generar los embeddings
        try:
            embedding = DeepFace.represent(img_path=img, model_name="Facenet512")
            embeddings.append(embedding[0]["embedding"])
            print(f"Embeddings generados para la imagen: {img}")
        except Exception as e:
            print(f"Error al procesar {img}: {e}")  # Registro de errores en la consola
    return np.array(embeddings)

if __name__ == "__main__":
    start_time = time.time()  # Iniciar temporizador

    # Cargar datos
    data_dir = os.path.join(os.path.dirname(__file__), '../data')  # Ruta relativa a la carpeta de datos
    images, labels = load_data(data_dir)
    load_time = time.time() - start_time  # Tiempo de carga de datos

    # Generar embeddings usando DeepFace.represent
    start_embedding_time = time.time()  # Iniciar temporizador para embeddings
    embeddings = generate_embeddings(images)
    embedding_time = time.time() - start_embedding_time  # Tiempo de generación de embeddings

    # Ruta relativa a la carpeta de modelos
    models_dir = os.path.join(os.path.dirname(__file__), '../models')

    # Guardar embeddings y etiquetas
    np.save(os.path.join(models_dir, 'embeddings.npy'), embeddings)
    np.save(os.path.join(models_dir, 'labels.npy'), labels)

    # Crear informe
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_images": len(images),
        "successful_embeddings": len(embeddings),
        "failed_images": len(images) - len(embeddings),
        "load_time_seconds": load_time,
        "embedding_time_seconds": embedding_time,
        "embeddings_shape": embeddings.shape if len(embeddings) > 0 else None,
        "model_used": "Facenet512",
        "embeddings_saved_path": os.path.abspath(os.path.join(models_dir, 'embeddings.npy')),
        "labels_saved_path": os.path.abspath(os.path.join(models_dir, 'labels.npy')),
        "persons_detected": {}
    }

    # Contar imágenes por persona
    for label in set(labels):
        report["persons_detected"][label] = labels.count(label)

    # Guardar informe en JSON
    report_path = os.path.join(models_dir, 'informe', 'generate_embeddings_report.json')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)  # Crear carpeta si no existe
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)

    print("Embeddings generados y guardados en 'models/'. Informe guardado en 'informe/generate_embeddings_report.json'")
