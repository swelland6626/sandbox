FROM tensorflow/tensorflow:2.6.0-gpu-jupyter

RUN python -m pip install --upgrade pip
COPY mnist_ex.py .