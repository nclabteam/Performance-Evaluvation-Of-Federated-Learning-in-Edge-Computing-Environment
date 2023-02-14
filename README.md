# Performance-Evaluation-Of-Federated-Learning-in-Edge-Computing-Environment

This code is developed to run Federated Learning application on Kubernetes version 1.21.0 and KubeEdge 1.10.0

* Step 1: Follow these two tutorial files to install Docker, Kubernets, KubeEdge and setup Federated Learning application in cloud node and edge node: TUTORIAL ON SETUP KUBEEDGE ENVIRONMENT ON CLOUD AND EDGE-1.pdf

* Step 2: Install CloudStream/EdgeStream on cloud node and edge node
* Step 3: Build docker file:
+ fashion-mnist-aggregator.Dockerfile
+ fashion-mnist-client.Dockerfile
Step 4: Check docker image on cloud node and edge node
Step 5: Run FL application
./run-federated.sh
Step 6: Check the training state on cloud node
kubectl logs -f fashion-mnist-cloud
Step 7: Stop FL application
./stop-federated.sh
