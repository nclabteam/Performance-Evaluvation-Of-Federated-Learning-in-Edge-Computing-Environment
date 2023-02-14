FROM tensorflow/tensorflow:2.6.0

RUN python -m pip install --upgrade pip
#RUN pip install tensorflow-addons

#RUN pip install virtualenv



####Don't need to install flwr and tensorflow anymore
#RUN pip install flwr 
#RUN pip install tensorflow

WORKDIR /home/work
COPY federated-learning/fashion-mnist-cloud_v1.0.8 /home/work/
#RUN export PYTHONPATH="${PYTHONPATH}:/lib/python3.8/site-packages/"

#RUN pip install -r requirements.txt

ENTRYPOINT ["python", "server.py"]

