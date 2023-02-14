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

"""Minimal example on how to start a simple Flower server."""
import time
import argparse
from typing import Callable, Dict, Optional, Tuple
from xmlrpc.client import Boolean

import numpy as np

import flwr as fl

import fashion_mnist

DEFAULT_SERVER_ADDRESS = "[::]:8080"

class NewStrategy(fl.server.strategy.FedAvg):       
    #Create list with 10 elements for 10 clients, each elements is also a list.
    local_accuracy_list = [ [], [], [], [], [], [], [], [], [], [] ] 
    local_loss_list = [ [], [], [], [], [], [], [], [], [], [] ] 
    local_training_time_list = [ [], [], [], [], [], [], [], [], [], [] ] 
    local_latency_list = [ [], [], [], [], [], [], [], [], [], [] ] 
    round_list = []
    summary_latency_list = [ [], [], [], [], [], [], [], [], [], [] ] 
    summary_throughput_list = [ [], [], [], [], [], [], [], [], [], [] ] 
    #Get the maximum round number, this function just only run once in the program
    def get_input_parameters(self, rounds, round_timeout, enable_latency, min_cpu, min_data):
        self.enable_latency = enable_latency
        self.rounds = rounds
        self.round_timeout = round_timeout
        self.min_cpu = min_cpu
        self.min_data = min_data

        
    #Get the client ID and append the local param(accuracy,loss,training_time) from all clients into lists
    def handle_client(self,hostname):
        self.client_idx = int(hostname[4:])
        self.local_accuracy_list[self.client_idx - 1].append(self.local_accuracy)
        self.local_loss_list[self.client_idx - 1].append(self.local_loss)
        self.local_training_time_list[self.client_idx - 1].append(self.local_training_time)
        self.local_latency_list[self.client_idx - 1].append(self.latency)
               

    #Get the correct_latency value and throughput for all clients in every round
    def calc_latency_throughput(self, training_time_list: list, latency_list: list, current_round: int):
        self.data_size = 6654740/1000000 #6.65MB (For MNIST and Fashion-MNIST dataset)
        
        #Remove empty client from local training_time_list if any
        time_filter_1 = [i for i in training_time_list if len(i) !=0]
        #Take the local training_time info with correspoding round
        time_filter_2 = [i[current_round-1] for i in time_filter_1]
        #Find the maximum local training time on the current round
        self.max_local_training_time= max(time_filter_2)

        #Remove empty client from latency list if any
        latency_filter_1 = [i for i in latency_list if len(i) !=0]
        #Take the latency info with correspoding round
        latency_filter_2 = [i[current_round-1] for i in latency_filter_1]

        self.correct_latency_list = [] 
        self.throughput_list = []
        for i in range(0, len(time_filter_2)): #len(time_filter_2) = len(latency_filter_2) = #of training clients, this loop calculate correct latency value for each client
            correct_latency = abs(time_filter_2[i] + latency_filter_2[i] - self.max_local_training_time)
            self.correct_latency_list.append(correct_latency)
            throughput = self.data_size/correct_latency
            self.throughput_list.append(throughput)
        for index in range(0,len(self.correct_latency_list)): #This loop plot latency and throughput from every clients at current round
            print(f"========================================================ROUND {current_round}===================================================")
            print(f"Latency from  edge{index+1}: {round(self.correct_latency_list[index],4)} seconds = {round(self.correct_latency_list[index]*1000,4)} ms")
            print(f"Throughput from  edge{index+1}: {round(self.throughput_list[index],4)} MB/sec") 


    def aggregate_fit(self,rnd,results, failures):
        agg_weights = super().aggregate_fit(rnd,results,failures)
        print(f"Maximum time for each round is: {self.round_timeout} seconds")
        
        
        for _,fit_res in results: #Each loop will handle 1 clients
            num_samples = fit_res.num_examples
            hostname = fit_res.metrics["hostname"]
            num_cpu = fit_res.metrics["cpu"]
            start_index = fit_res.metrics["start_index"]
            end_index = fit_res.metrics["end_index"]
            ###Get the data_size value here
            

            if (num_cpu >= self.min_cpu) and (num_samples >= self.min_data):
                self.local_loss =  round(fit_res.metrics["loss"],4)
                self.local_accuracy = round(fit_res.metrics["accuracy"],4)
                self.local_training_time = fit_res.metrics["training_time"]
                
                client_checkpoint = fit_res.metrics["client_checkpoint"]
                cloud_checkpoint = time.time()
                self.latency = cloud_checkpoint - client_checkpoint

                print(f"========================================================ROUND {rnd}===================================================")
                print(f"Results from {hostname} (from: {start_index} ~ {end_index} = {num_samples} samples) - ({num_cpu} CPU)- Accuracy: {self.local_accuracy} - Loss: {self.local_loss} - Local training time: {round(self.local_training_time,4)} seconds")


                #This function will be called on every round for all available client in the system
                self.handle_client(hostname)
    
        #Get the round number and append to a list
        self.round_list.append(rnd)
      
       #Return the latency/throughput infomation and sum all the results(latency,throughput) into a list
        if (self.enable_latency == 1):
            self.calc_latency_throughput(training_time_list = self.local_training_time_list, latency_list = self.local_latency_list, current_round = rnd)
            for i in range(0,len(self.correct_latency_list)):
                self.summary_latency_list[i].append(self.correct_latency_list[i])
                self.summary_throughput_list[i].append(self.throughput_list[i])


        
        if (rnd == self.rounds): #print the summary results at the final round
            for i in range(0, 10):
                if len(self.local_accuracy_list[i]) == 0: #If one of 10 clients didn't join the training
                    print(f"Edge {i+1} didn't join the training process")
                else:
                    total_zip_accuracy_results = tuple(zip(self.round_list, self.local_accuracy_list[i]))
                    total_zip_loss_results = tuple(zip(self.round_list, self.local_loss_list[i]))
                    total_zip_training_time_results = tuple(zip(self.round_list, self.local_training_time_list[i]))

                    print(f"========================================================SUMMARY - ACCURACY - LOSS - TRAINING TIME - EDGE {i+1}===================================================")
                    print(f"SUMMARY ACCURACY from edge {i+1}: {total_zip_accuracy_results}")
                    print(f"SUMMARY LOSS from edge {i+1}: {total_zip_loss_results}")
                    print(f"SUMMARY TRAINING TIME from edge {i+1}: {total_zip_training_time_results}")  
                    if (self.enable_latency == 1):
                        total_zip_latency_results = tuple(zip(self.round_list, self.summary_latency_list[i]))
                        total_zip_throughput_results = tuple(zip(self.round_list, self.summary_throughput_list[i]))
                        print(f"SUMMARY LATENCY from edge {i+1}: {total_zip_latency_results}")
                        print(f"SUMMARY THROUGHPUT from edge {i+1}: {total_zip_throughput_results}")  
        return agg_weights

def main() -> None:
    """Start server and train five rounds."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--server_address",
        type=str,
        default=DEFAULT_SERVER_ADDRESS,
        help=f"gRPC server address (default: {DEFAULT_SERVER_ADDRESS})",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of rounds of federated learning (default: 1)",
    )
    parser.add_argument(
        "--sample_fraction",
        type=float,
        default=0.1,
        help="Fraction of available clients used for fit/evaluate (default: 0.1)",
    )
    parser.add_argument(
        "--min_sample_size",
        type=int,
        default=1,
        help="Minimum number of clients used for fit/evaluate (default: 1)",
    )
    parser.add_argument(
        "--min_num_clients",
        type=int,
        default=1,
        help="Minimum number of available clients required for sampling (default: 1)",
    )

    parser.add_argument(
        "--log_host",
        type=str,
        help="Logserver address (no default)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--min_data",
        type=int,
        default = 1000,
        help="Minimum number of training samples requirements",
    )
    parser.add_argument(
        "--min_cpu",
        type=int,
        default=1,
        help="Minimum number of cpu requirements"
    )
    parser.add_argument(
        "--round_timeout",
        type=int,
        help="Time for each client do the training tasks,the results will be discard if any client take more time than round_timeout (seconds)",
        default=30,
    )
    parser.add_argument(
        "--enable_latency_capture",
        type=int,
        help="Enable the latency/throughput measurement (all clients need to train at the same time), do not allow any client was dropped during the training",
        default=0,
    )
    args = parser.parse_args()

    # Load evaluation data
    _, xy_test = fashion_mnist.load_data(start_index=0, end_index=1, class_type=10, client_id = 1, data_type= "IID")
    
    def fit_config(rnd: int) -> Dict[str, fl.common.Scalar]:
        """Return a configuration with static batch size and (local) epochs."""
        ###All clients will receive these below information in each round
        config: Dict[str, fl.common.Scalar] = {
            "round": str(rnd),
            "epochs": str(args.epochs),
            "batch_size": str(args.batch_size),
        }
        return config
    # Create strategy
    strategy = NewStrategy(
        fraction_fit=args.sample_fraction,
        min_fit_clients=args.min_sample_size,
        min_eval_clients= args.min_num_clients, #number of clients used during training will equal to number of clients used during validation.
        min_available_clients=args.min_num_clients,
        eval_fn=get_eval_fn(xy_test=xy_test),
        on_fit_config_fn=fit_config,
    )
    strategy.get_input_parameters(enable_latency = args.enable_latency_capture,rounds = args.rounds, round_timeout = args.round_timeout, min_data= args.min_data, min_cpu=args.min_cpu)
    
    # Configure logger and start server
    fl.common.logger.configure("server", host=args.log_host)
    fl.server.start_server(
        args.server_address,
        config={"num_rounds": args.rounds, "round_timeout": args.round_timeout},
        strategy=strategy,   
    )



def get_eval_fn(
    xy_test: Tuple[np.ndarray, np.ndarray]
) -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        """Use the entire MNIST test set for evaluation."""
        model = fashion_mnist.load_model()
        model.set_weights(weights)
        loss, acc = model.evaluate(xy_test[0], xy_test[1], batch_size=len(xy_test))
        return float(loss), float(acc)

    return evaluate


if __name__ == "__main__":
    main()

