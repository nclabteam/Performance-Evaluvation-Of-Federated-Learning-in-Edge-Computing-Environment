# Performance-Evaluation-Of-Federated-Learning-in-Edge-Computing-Environment

This code is developed to run Federated Learning application on Kubernetes version 1.21.0 and KubeEdge 1.10.0

* Step 1: Follow these two tutorial files to install Docker, Kubernets, KubeEdge and setup Federated Learning application in cloud node and edge node: TUTORIAL ON SETUP KUBEEDGE ENVIRONMENT ON CLOUD AND EDGE-1.pdf & Federated Learning Docker Tutorial-1.pdf

* Step 2: Clone the repository by pasting the below command in the terminal.
  ````
  https://github.com/nclabteam/Performance-Evaluvation-Of-Federated-Learning-in-Edge-Computing-Environment.git
  ````

* Step 3: After the repository is cloned, open the cloned repository in terminal.
  ````
  cd Performance-Evaluvation-Of-Federated-Learning-in-Edge-Computing-Environment/fashion-mnist/federated-learning/
  ````
* Step 4: Install the project requirements or dependencies inside virtual environment for both fashion-mnist-client and fashion-mnist-cloud folder using terminal as:
  ````
  pip3 install -r requirements.txt
  ````
* Step 5: Config the cloud.yaml and client.yaml based on the number of nodes and the input parameters.
  ```` 
  cd Performance-Evaluvation-Of-Federated-Learning-in-Edge-Computing-Environment/fashion-mnist/yaml
  ````
* Step 6: Deploy the Federated Learning application
  ````
  ./run-federated.sh
  ````
* Step 7: Check the training state on cloud node
  ````
  kubectl logs -f fashion-mnist-cloud
  ````
* Step 8: Stop FL application
  ````
  ./stop-federated.sh
  ````
