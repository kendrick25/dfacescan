Pasos para instalación de dfsd

Requisitos

python 3.8

librerías

pytorch torchvision torchaudio cudatoolkit opencv

conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch  

conda install conda-forge::opencv

Instalar el repo

pip install git+https://github.com/hukkelas/DSFD-Pytorch-Inference.git

Para correr el programa se necesita el test.py y un directorio llamado images/ con las imagenes a procesar

Las imagenes procesadas se almacenaran en el mismo directorio

Para ejecutar test.py

python3 test.py
