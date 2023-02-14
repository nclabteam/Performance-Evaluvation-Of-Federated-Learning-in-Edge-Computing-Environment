# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower client example using TensorFlow for FASHION-MNIST image classification."""
import os
from logging import INFO
import argparse
from typing import Dict, Tuple, cast

import numpy as np
import tensorflow as tf

import flwr as fl
from flwr.common import Weights

import fashion_mnist
import time
from flwr.common.logger import log

import socket
import random
DEFAULT_SERVER_ADDRESS = "[::]:8080"
Round = 0
round_list = []
training_time_list = []
loss_list = []
acc_list = []
class FashionMnistClient(fl.client.KerasClient):
    """Flower KerasClient implementing FASHION-MNIST image classification."""
    
    def __init__(
        self,
        model: tf.keras.Model,
        xy_train: Tuple[np.ndarray, np.ndarray],
        xy_test: Tuple[np.ndarray, np.ndarray],
    ):
        self.model = model
        self.x_train, self.y_train = xy_train
        self.x_test, self.y_test = xy_test
    def get_input_parameters(self, client_id, top_selection, start_index, end_index):
        self.client_id = client_id
        self.top_selection = top_selection
        ###Use only in statistic heterogeneity
        self.start_index = start_index
        self.end_index = end_index
    def get_weights(self) -> Weights:
        return cast(Weights, self.model.get_weights())
    
    def fit(
        self, weights: Weights, config: Dict[str, fl.common.Scalar]
    ) -> Tuple[Weights, int, int, dict]:
        
        # Use provided weights to update local model
        self.model.set_weights(weights)

        #Take the round number
        round = int(config['round']) 
        
        """if (self.top_selection == 20 and self.client_id == 1) or (self.top_selection == 40 and self.client_id == 1) or (self.top_selection == 60 and self.client_id == 1) or (self.top_selection == 80 and self.client_id == 1) or (self.top_selection == 100 and self.client_id == 1): 
            #Top 20-40-60-80-100-edge 1
            #Calculate the start-end index of training samples
            round = round - 1
            step = round % 10 #First round: 0 ~ 3000

        elif (self.top_selection == 20 and self.client_id == 6) or (self.top_selection == 40 and self.client_id == 6) or (self.top_selection == 60 and self.client_id == 6) or (self.top_selection == 80 and self.client_id == 6) or (self.top_selection == 100 and self.client_id == 6): 
            #Top 20-40-60-80-100-edge 6
            #Calculate the start-end index of training samples 
            round = round - 1
            step = (round+1) % 10 #First round: 3000~6000

        elif (self.top_selection == 40 and self.client_id == 2) or (self.top_selection == 60 and self.client_id == 2) or (self.top_selection == 80 and self.client_id == 2) or (self.top_selection == 100 and self.client_id == 2): 
            #Top 40-60-80-100-edge 2
            #Calculate the start-end index of training samples 
            round = round - 1
            step = (round+2) % 10 #First round: 6000~9000    

        elif (self.top_selection == 40 and self.client_id == 7) or (self.top_selection == 60 and self.client_id == 7) or (self.top_selection == 80 and self.client_id == 7) or (self.top_selection == 100 and self.client_id == 7): 
            #Top 40-60-80-100-edge 7
            #Calculate the start-end index of training samples 
            round = round - 1
            step = (round+3) % 10 #First round: 9000~12000

        elif (self.top_selection == 60 and self.client_id == 3) or (self.top_selection == 80 and self.client_id == 3) or (self.top_selection == 100 and self.client_id == 3): 
            #Top 60-80-100-edge 3
            #Calculate the start-end index of training samples 
            round = round - 1
            step = (round+4) % 10 #First round: 12000~15000

        elif (self.top_selection == 60 and self.client_id == 8) or (self.top_selection == 80 and self.client_id == 8) or (self.top_selection == 100 and self.client_id == 8): 
            #Top 60-80-100-edge 8
            #Calculate the start-end index of training samples 
            round = round - 1
            step = (round+5) % 10 #First round: 15000~18000
        
        elif (self.top_selection == 80 and self.client_id == 4) or (self.top_selection == 100 and self.client_id == 4): 
            #Top 80-100-edge 4
            #Calculate the start-end index of training samples 
            round = round - 1
            step = (round+6) % 10 #First round: 18000~21000

        elif (self.top_selection == 80 and self.client_id == 9) or (self.top_selection == 100 and self.client_id == 9): 
            #Top 80-100-edge 9
            #Calculate the start-end index of training samples 
            round = round - 1
            step = (round+7) % 10 #First round: 21000~24000
        
        elif (self.top_selection == 100 and self.client_id == 5):
            #Top 100-edge 5
            #Calculate the start-end index of training samples 
            round = round - 1
            step = (round+8) % 10 #First round: 24000~27000

        elif (self.top_selection == 100 and self.client_id == 10):
            #Top 100-edge 10
            #Calculate the start-end index of training samples 
            round = round - 1
            step = (round+9) % 10 #First round: 27000~30000 """
        
        
        
        #start = step * 3000
        #end = start + 3000
        ###Modify code here to randomly activate the data distribution ratio, from [0:30000]
        ###Fit fucntion will apply only when you want to change data distribution in each round



        """ print(f"Start : {start} End : {end}")
        x_train = self.x_train[start:end]
        y_train = self.y_train[start:end] """
        
        print(f"==========================================ROUND {round}====================================== ")
        start_time = time.time()
        # Train the local model using local dataset
        self.model.fit(
            self.x_train,
            self.y_train,
            batch_size=int(config["batch_size"]),
            epochs=int(config["epochs"]),
        )

        end_time = time.time()
        training_time = end_time - start_time
        
        #Return updated model parameters and results
        round_list.append(round+1)
        training_time_list.append(training_time)
        history_training_time =  tuple(zip (round_list,training_time_list))        
        print(f"Local training time: {history_training_time}")
        print(f"Local model performance at round {round+1}: ")
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, batch_size=int(config["batch_size"]), verbose=2)
        loss_list.append(loss)
        acc_list.append(accuracy)
        
        history_loss = tuple(zip (round_list,loss_list)) 
        history_acc = tuple(zip (round_list,acc_list)) 
        
        print(f"Local history loss: {history_loss}")
        print(f"Local history accuracy: {history_acc}")
        #Get hostname
        hostname = socket.gethostname()

        client_send_checkpoint = time.time()
        #Get the number of CPU:
        num_cpu = os.cpu_count()
        
        results = {
            "hostname":hostname,
            "loss": loss,
            "accuracy": accuracy,
            "training_time": training_time,
            "client_checkpoint": client_send_checkpoint,
            "cpu": num_cpu,
            "start_index": self.start_index,
            "end_index": self.end_index,
        }
        
        
        # Return the refined weights and the number of examples used for training
        return self.model.get_weights(), len(self.x_train), len(self.x_train), results

    def evaluate(
        self, weights: Weights, config: Dict[str, fl.common.Scalar]
    ) -> Tuple[int, float, float]:
        # Update local model and evaluate on local dataset
        self.model.set_weights(weights)
        print("Global model performance: ")
        loss, accuracy = self.model.evaluate(
            self.x_test, self.y_test, batch_size=len(self.x_test), verbose=2
        )

        # Return number of evaluation examples and evaluation result (loss/accuracy)
        return len(self.x_test), float(loss), float(accuracy)


def main() -> None:
    """Load data, create and start Fashion-MnistClient."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--server_address",
        type=str,
        default=DEFAULT_SERVER_ADDRESS,
        help=f"gRPC server address (default: {DEFAULT_SERVER_ADDRESS})",
    )
    parser.add_argument(
        "--client_id", type=int, required=True, help="Client ID"
    )
    parser.add_argument(
        "--start_index", type=int, required=True, help="start_index index (no default)"
    )
    parser.add_argument(
        "--end_index",
        type=int,
        required=True,
        help="Number of clients (no default)",
    )
    parser.add_argument(
        "--class_type", type=str, required=True, help="Type of class (0~9) category"
    )
    parser.add_argument(
        "--data_type", type=str, required=True, help="Data type: IID, 1class-non-IID, 2class-non-IID, 10class-non-IID"
    )
    parser.add_argument(
        "--top_selection", type=int, default=100, help="Choosing top client selection, 100-80-60-40-20"
    )
    parser.add_argument(
        "--log_host",
        type=str,
        help="Logserver address (no default)",
    )
    args = parser.parse_args()
    
    # Configure logger
    fl.common.logger.configure(f"client_{args.start_index}", host=args.log_host)

    
    # Load model and data
    model = fashion_mnist.load_model()
    

    """ ###Randomly distribute data for each client for statistic heterogeneity
    random_factor = random.randint(0,30)
    if args.client_id %2 != 0: #Odd client
        start = 0
        end = random_factor*1000
    if args.client_id %2 == 0: #Even client
        start = random_factor*1000
        end = 30000 """


    xy_train, xy_test = fashion_mnist.load_data(
        start_index=args.start_index, end_index=args.end_index, client_id=args.client_id, class_type=args.class_type, data_type=args.data_type
    )

    #xy_train, xy_test = fashion_mnist.load_data(
    #    start_index=args.start_index, end_index=args.end_index, client_id=args.client_id, class_type=args.class_type, data_type=args.data_type
    #)

    # Start client
    client = FashionMnistClient(model, xy_train, xy_test)

    client.get_input_parameters(client_id=args.client_id, top_selection= args.top_selection,start_index=args.start_index,end_index=args.end_index)

    fl.client.start_keras_client(args.server_address, client)



if __name__ == "__main__":
    main()
