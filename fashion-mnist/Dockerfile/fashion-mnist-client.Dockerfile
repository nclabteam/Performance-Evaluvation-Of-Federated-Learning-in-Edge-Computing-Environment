FROM tensorflow/tensorflow:2.6.0

RUN python -m pip install --upgrade pip

#RUN pip install tensorflow-addons
#RUN pip install tensorflow

WORKDIR /home/work
COPY federated-learning/fashion-mnist-client_v1.0.8 /home/work/

ENTRYPOINT ["python","client.py"]
