# Entrenamiento de modelo con estrucura para embeddings entrenado con FaceNet512
 
Este proyecto utiliza el modelo preentrenado FaceNet512 para generar embeddings faciales y un clasificador RandomForestClassifier para reconocer rostros.

## Estructura del Proyecto
```txt
entrenamiento_model/
├── data/
│   ├── scarlett-johansson/
│   │   ├── scarlett-johansson_rostro_0.jpg
│   │   ├── scarlett-johansson_rostro_1.jpg
│   └── Thais/
│       ├── Thais_rostro_0.jpg
│       └── Thais_rostro_1.jpg
├── models/
└── src/
│   ├── load_data.py
│   ├── generate_embeddings.py
│   ├── train_classifier.py
│   └── analyze_results.py
```
- `data/`: Contiene las imágenes de rostros organizadas en carpetas por persona.
- `src/`: Contiene los scripts para cargar datos, generar embeddings y entrenar el clasificador.
- `models/`: Almacena el clasificador entrenado y el codificador de etiquetas.

## Instrucciones

1. Coloca las imágenes en la carpeta `data/` en subcarpetas organizadas por persona, puedes revisar la seccion de   create data para guiarte de como obtener una data, con la estructura necesaria para el entrenamiento, solo crea la data y muevela a esta ruta actual de  `data/` para realizar el entrenamiento.

2. Ejecuta los scripts en este orden:
- `Primero`: load_data.py (opcional para verificación).
- `Segundo`: generate_embeddings.py (para generar y guardar embeddings).
- `Tercero`: train_classifier.py (para entrenar el clasificador y guardar el modelo).
- `(opcional) Cuarto`: analyze_results.py (para obtener un informe de los resultados del entrenamiento).

## Notas
- Asegúrate de que las imágenes están bien organizadas.
- El modelo preentrenado FaceNet512 es utilizado para la generación de embeddings.
