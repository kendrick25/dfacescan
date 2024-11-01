## Estructura de carpetas necesarias para el la deteción de forma comparativa con Facenet512 y RetinaFace
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
scikit-learn
scipy
matplotlib
imutils
joblib
```
#### Ejecutar en consola, para la instalación de las dependencias:
>[!NOTE] 
>Estamos trabajando en la version:
>```powershell
>Python 3.10.15
>```
```powershell
pip install -r requirements.txt
```
###### Verifica los paquetes instalados:
```powershell
python versions.py
```
