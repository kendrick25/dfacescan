# Face Detection

##### Tested techniques

* OpenCV Haar Cascades

* Multi-task Cascaded Convolutional Neural Network (MTCNN)

* Retina Face

##### Non-Tested techniques

* PyramidBox

* DSFD (Dual Shot Face Detector)

* FaceBoxes (Real Time)

##### Environment

```bash
--Create Environment
conda create --name faceretina python=3.10.8
python -m pip install -U pip
pip install numpy==1.26.4
pip install sympy
pip install pillow
pip install matplotlib
pip install opencv-python
python -m pip install -U matplotlib
pip install mtcnn
conda install -c conda-forge notebook
pip install retina-face

--Tensorflow
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
python -m pip install "tensorflow<2.11"
```
