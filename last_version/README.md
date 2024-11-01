## Estructura de carpetas necesarias para el la deteción de forma comparativa
```txt
prueba_deepface/
├── data/
│   ├── scarlett-johansson/
│   │   ├── scarlett-johansson_rostro_0.jpg
│   │   ├── scarlett-johansson_rostro_1.jpg
│   └── Thais/
│       ├── Thais_rostro_0.jpg
│       └── Thais_rostro_1.jpg
├── test/
│   └── img-00.png
└── result/   # Esta carpeta debe existir para guardar los resultados
```
## Instalación de dependencias
#### requirements.txt:
```txt
deepface==0.0.93
opencv-python==4.10.0.84
numpy==1.26.4
tensorflow==2.10.1
tensorflow-cpu==2.10.1
retina-face==0.0.17
scipy
matplotlib
imutils
```
#### Ejecutar en consola:
```powershell
pip install -r requirements.txt
```

>[!NOTE] 
>Estamos trabajando en la version:
>```powershell
>Python 3.10.15
>```

>[!TIP] 
>Estamos trabajando en la version:
>```powershell
>Python 3.10.15
>```

>[!IMPORTANT] 
>Estamos trabajando en la version:
>```powershell
>Python 3.10.15
>```

>[!WARNING] 
>Estamos trabajando en la version:
>```powershell
>Python 3.10.15
>```

>[!CAUTION] 
>Estamos trabajando en la version:
>```powershell
>Python 3.10.15
>```
## Parametros del codigo a tomar en cuenta
```python
# Realizar la comparación entre el rostro detectado y el rostro objetivo
result = DeepFace.verify(
    img1_path=bgr_target_face,
    img2_path=bgr_face,
    detector_backend="skip",  # Omitir la detección ya que están los rostros
    model_name="Facenet512",
    distance_metric="cosine",
    threshold=0.49, # Ajustar este valor de aceptación de detección
    enforce_detection=True
)
```
