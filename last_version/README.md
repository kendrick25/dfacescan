# Instalación de dependencias
Si deseas trabajar con una tarjeta grafica dedicada instala las dependencias de CUDA:

```powershell
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```
>[!NOTE] 
>En algunos casos pueden variar las librerias de CUDA dependiendo de la tarjeta grafica

```powershell
pip install -r requirements.txt
```
si quieres trabajar solo con el CPU:

```powershell
tensorflow-cpu==2.10.1
```

#### requirements.txt:
```txt
deepface==0.0.93
opencv-python==4.10.0.84
numpy==1.26.4
tensorflow==2.10.1
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
