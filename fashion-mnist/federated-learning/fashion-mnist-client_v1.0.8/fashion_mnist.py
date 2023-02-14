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
"""Flower client example using TensorFlow for Fashion-MNIST image classification."""


from typing import Tuple, cast

import os
import gzip
import numpy as np
import tensorflow as tf
from keras.utils.data_utils import get_file

tf.get_logger().setLevel("ERROR")

SEED = 2020

#Get the current directory to the local datasets
current_dir = os.getcwd() +'/datasets'


def load_model(input_shape: Tuple[int, int, int] = (28, 28, 1)) -> tf.keras.Model:
    """Load model for Fashion-MNIST."""
    # Kernel initializer
    kernel_initializer = tf.keras.initializers.glorot_uniform(seed=SEED)

    # Architecture
    inputs = tf.keras.layers.Input(shape=input_shape)
    layers = tf.keras.layers.Conv2D(
        32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer=kernel_initializer,
        padding="same",
        activation="relu",
    )(inputs)
    layers = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(layers)
    layers = tf.keras.layers.Conv2D(
        64,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer=kernel_initializer,
        padding="same",
        activation="relu",
    )(layers)
    layers = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(layers)
    layers = tf.keras.layers.Flatten()(layers)
    layers = tf.keras.layers.Dense(
        512, kernel_initializer=kernel_initializer, activation="relu"
    )(layers)

    outputs = tf.keras.layers.Dense(
        10, kernel_initializer=kernel_initializer, activation="softmax"
    )(layers)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=["accuracy"],
    )
    #model.summary()
    return model


def load_data(
    start_index: int, end_index: int, client_id: int, class_type: str, data_type: str
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load partition of randomly shuffled Fashion-MNIST subset."""
    # Load training and test data from local 
    files = [
      'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
      't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]
    
    paths = []
    for fname in files:
        paths.append(get_file(fname, origin =  fname, cache_subdir = current_dir))
    
    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)
    #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    print('x_train shape (full): ', x_train.shape)
    print('y_train shape (full): ', y_train.shape)
    print('X_test shape (full): ', x_test.shape)
    print('y_test shape (full): ', y_test.shape)
    
    if (data_type == "IID"):
        #IID data
        print("#########IID DATA DISTRIBUTION###########")
        x_train, y_train = get_partition_test(x_train, y_train, start_index, end_index)
        
    if (data_type == "1class-non-IID"):
        #1-class non-IID data
        print("#########1 CLASS NON-IID DATA DISTRIBUTION###########")
        class_type_int = int(class_type)
        x_train = x_train[y_train == class_type_int]
        y_train = y_train[y_train == class_type_int]
        x_train = x_train[:3000]
        y_train = y_train[:3000]
        #x_train, y_train = get_partition_1_class((x_train, y_train,class_type_int)
        
    if (data_type == "2class-non-IID"):
        #2-class non-IID data
        print("#########2 CLASS NON-IID DATA DISTRIBUTION###########") 
        if (client_id%2 != 0): #Odd clients
           class_type_split = class_type.split(",")
           #All training samples of the first class
           x_train_1st_class =  x_train[y_train == int(class_type_split[0])]
           y_train_1st_class =  y_train[y_train == int(class_type_split[0])]
           #Take only 3000 samples of the first class
           x_train_1st_class =  x_train_1st_class[:1500] 
           y_train_1st_class =  y_train_1st_class[:1500] 
           
           #All training samples of the second class
           x_train_2nd_class =  x_train[y_train == int(class_type_split[1])]
           y_train_2nd_class =  y_train[y_train == int(class_type_split[1])]
           #Take only 3000 samples of the second class
           x_train_2nd_class = x_train_2nd_class[:1500]
           y_train_2nd_class = y_train_2nd_class[:1500]
           
           #Combine these 2 part of samples
           x_train = np.concatenate((x_train_1st_class,x_train_2nd_class))
           y_train = np.concatenate((y_train_1st_class,y_train_2nd_class))
           
        else:#Even clients
           class_type_split = class_type.split(",")
           #All training samples of the first class
           x_train_1st_class =  x_train[y_train == int(class_type_split[0])]
           y_train_1st_class =  y_train[y_train == int(class_type_split[0])]
           #Take the rest samples of the first class
           x_train_1st_class =  x_train_1st_class[1500:3000] 
           y_train_1st_class =  y_train_1st_class[1500:3000] 
           
           #All training samples of the second class
           x_train_2nd_class =  x_train[y_train == int(class_type_split[1])]
           y_train_2nd_class =  y_train[y_train == int(class_type_split[1])]
           #Take the rest samples of the second class
           x_train_2nd_class = x_train_2nd_class[1500:3000]
           y_train_2nd_class = y_train_2nd_class[1500:3000]
           
           #Combine these 2 part of samples
           x_train = np.concatenate((x_train_1st_class,x_train_2nd_class))	
           y_train = np.concatenate((y_train_1st_class,y_train_2nd_class))
    
    print("Label: ", y_train)
    
    # Shuffle the training data
    x_train, y_train = shuffle(x_train, y_train, seed=SEED)
    x_test, y_test = shuffle(x_test, y_test, seed=SEED)
    print("Label_shuffled: ", y_train)
    #x_train, y_train = get_partitionv4(x_train, y_train, partition, num_partitions) ###Uncomment this to test fedsim.py, test on 2 clients
    #x_train, y_train = get_partitionv3(x_train, y_train, partition) ###Uncomment this line to test train.py, test on 1 client
    
    #x_train, y_train = get_partitionv2(x_train, y_train, partition) ###Uncomment both lines when test server-client flower project
    #x_test, y_test = get_partition(x_test, y_test, partition, num_partitions)
    
    print('x_train shape: ', x_train.shape)
    print('y_train shape: ', y_train.shape)
    print('x_test shape: ', x_test.shape)
    print('y_test shape: ', y_test.shape)
    
    # Adjust x sets shape for model
    x_train = adjust_x_shape(x_train)
    x_test = adjust_x_shape(x_test)

    # Normalize data
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    
    # Convert class vectors to one-hot encoded labels
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


def adjust_x_shape(nda: np.ndarray) -> np.ndarray:
    """Turn shape (x, y, z) into (x, y, z, 1)."""
    nda_adjusted = np.reshape(nda, (nda.shape[0], nda.shape[1], nda.shape[2], 1))
    return cast(np.ndarray, nda_adjusted)


def shuffle(
    x_orig: np.ndarray, y_orig: np.ndarray, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Shuffle x and y in the same way."""
    np.random.seed(seed)
    idx = np.random.permutation(len(x_orig))
    #print(idx[:20])
    return x_orig[idx], y_orig[idx]
    
    
def get_partition_test(
    x_orig: np.ndarray, y_orig: np.ndarray, start_index: int, end_index: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a single partition of an equally partitioned dataset."""
    print("Start index traning image: %d" %start_index)
    print("End index training image: %d" %end_index)
    return x_orig[start_index:end_index], y_orig[start_index:end_index]

# def get_partition(
    # x_orig: np.ndarray, y_orig: np.ndarray, partition: int, num_clients: int
# ) -> Tuple[np.ndarray, np.ndarray]:
    # """Return a single partition of an equally partitioned dataset."""
    # step_size = len(x_orig) / num_clients
    # start_index = int(step_size * partition)
    # end_index = int(start_index + step_size)
    # return x_orig[start_index:end_index], y_orig[start_index:end_index]
# #    return x_orig, y_orig

# def get_partitionv2(
    # x_orig: np.ndarray, y_orig: np.ndarray, partition: int
# ) -> Tuple[np.ndarray, np.ndarray]:
    # """Return a single partition of an equally partitioned dataset."""
    
    # #Split_number= int((len(x_orig)*partition*10) / 100)
    # if partition == 0:
        # return x_orig[:2000], y_orig[:2000]
    # if partition == 1:  
        # return x_orig[2000:3000], y_orig[2000:3000]
    # # if partition == 2:
        # # return x_orig[:60000], y_orig[:60000]    
    # #step_size = len(x_orig) / num_clients
    # #start_index = int(step_size * partition)
    # #end_index = int(start_index + step_size)
    
# def get_partitionv3(
    # x_orig: np.ndarray, y_orig: np.ndarray, partition: int
# ) -> Tuple[np.ndarray, np.ndarray]:
    # """Return a single partition of an equally partitioned dataset."""   
    # #train_sample = np.array([1000,5000,10000])
    
    # return x_orig[:partition], y_orig[:partition]
 
# def get_partitionv4(
    # x_orig: np.ndarray, y_orig: np.ndarray, client_id: int, num_samples: int
 # ) -> Tuple[np.ndarray, np.ndarray]:
    # """Return a single partition of an equally partitioned dataset."""
    # return x_orig[client_id*num_samples:(client_id+1)*num_samples], y_orig[client_id*num_samples:(client_id+1)*num_samples]
# #    return x_orig, y_orig

# def get_partitionv5(
    # x_orig: np.ndarray, y_orig: np.ndarray, client_id: int, num_samples: int
 # ) -> Tuple[np.ndarray, np.ndarray]:
    # """Return a single partition of an equally partitioned dataset."""
    # x_train_T_shirt           = x_train[y_train == 0]
    # x_train_Trouser          = x_train[y_train == 1]
    # return x_orig[client_id*num_samples:(client_id+1)*num_samples], y_orig[client_id*num_samples:(client_id+1)*num_samples]
    
