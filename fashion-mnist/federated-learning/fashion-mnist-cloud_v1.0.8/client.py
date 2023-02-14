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

    def get_weights(self) -> Weights:
        return cast(Weights, self.model.get_weights())
    
    def fit(
        self, weights: Weights, config: Dict[str, fl.common.Scalar]
    ) -> Tuple[Weights, int, int, dict]:
        global Round
        # Use provided weights to update local model
        self.model.set_weights(weights)
        # Get the round number
        count = Round + 1
        Round = count
        print(f"==========================================ROUND {count}====================================== ")
        start_time = time.time()
        # Train the local model using local dataset
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size=int(config["batch_size"]),
            epochs=int(config["epochs"]),
            #Round=int(config["rounds"]),
        )
        end_time = time.time()
        training_time = end_time - start_time
        
        #Return updated model parameters and results
        round_list.append(count)
        training_time_list.append(training_time)
        history_training_time =  tuple(zip (round_list,training_time_list))        
        print(f"Local training time: {history_training_time}")
        print(f"Local model performance at round {count}: ")
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
        
        results = {
            "hostname":hostname,
            "loss": loss,
            "accuracy": accuracy,
            "training_time": training_time,
            "client_checkpoint": client_send_checkpoint,
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
    """Load data, create and start MnistClient."""
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
        "--log_host",
        type=str,
        help="Logserver address (no default)",
    )
    args = parser.parse_args()
    
    # Configure logger
    fl.common.logger.configure(f"client_{args.start_index}", host=args.log_host)
    #log(INFO, "Starting client, ID: %s", args.start_index)
    #print("Edge 1")

    # Load model and data
    model = fashion_mnist.load_model()
    xy_train, xy_test = fashion_mnist.load_data(
        start_index=args.start_index, end_index=args.end_index, client_id=args.client_id, class_type=args.class_type, data_type=args.data_type
    )
    
    
    #log(INFO,"x_train shape: %s", xy_train[0].shape)
    #print('test log')
    #print('x_train shape: ' + xy_train[0].shape)
    #print('y_train shape: ', xy_train[1].shape)
    #print('x_test shape: ', xy_test[0].shape)
    #print('y_test shape: ', xy_test[1].shape)
    
    # Start client
    client = FashionMnistClient(model, xy_train, xy_test)
    fl.client.start_keras_client(args.server_address, client)



if __name__ == "__main__":
    main()
