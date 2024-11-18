## Estructura de carpetas necesarias para el la deteción con el uso del modelo
Este modelo fue  preentrenado con FaceNet512 para generar embeddings faciales y un clasificador RandomForestClassifier para reconocer rostros, el proceso de entrenamiento se puede ver en la carpeta `entrenamiento_model`.
```txt
prueba_deepface/
├── models/  # Modelo entrenado
│   ├── face_classifier.pkl
│   └── label_encoder.pkl
├── result/ # Carpeta para la imagen de salida
│   └── img-00.png
└── test/   # Carpeta para la imagen de entrada
│   └── img-00.png
└── test_models.py
```

#### Ejecutar en consola, para la instalación de las dependencias:
Revisar los requerimientos en la carpeta `last_version` si se presenta algun problema
>[!NOTE] 
>Estamos trabajando en la version:
>```powershell
>Python 3.10.15
>```

```powershell
python test_models.py
```
