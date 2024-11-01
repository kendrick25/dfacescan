# Face Recognition Project
 
Este proyecto utiliza el modelo preentrenado FaceNet512 para generar embeddings faciales y un clasificador SVM para reconocer rostros.

## Estructura del Proyecto
- `data/`: Contiene las imágenes de rostros organizadas en carpetas por persona.
- `src/`: Contiene los scripts para cargar datos, generar embeddings y entrenar el clasificador.
- `models/`: Almacena el clasificador entrenado y el codificador de etiquetas.

## Instrucciones

1. Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```

2. Coloca las imágenes en la carpeta `data/` en subcarpetas organizadas por persona.

3. Ejecuta los scripts:
Primero: load_data.py (opcional para verificación).
Segundo: generate_embeddings.py (para generar y guardar embeddings).
Tercero: train_classifier.py (para entrenar el clasificador y guardar el modelo).

## Notas
- Asegúrate de que las imágenes están bien organizadas.
- El modelo preentrenado FaceNet512 es utilizado para la generación de embeddings.
