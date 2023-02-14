#!/bin/bash

# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#ssss
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

set -e

SERVER_ADDRESS="192.168.1.4:8888"
NUM_CLIENTS=1

echo "Starting $NUM_CLIENTS clients."
for ((i = 0; i < $NUM_CLIENTS; i++))
do
    echo "Starting client(cid=$i) with partition $i out of $NUM_CLIENTS clients."
    curtime=$(date +'%Y%m%d_%H%M%S')
    python3 client.py \
	  --start_index=0 \
	  --end_index=6000 \
	  --client_id=1 \
	  --class_type="0"  \
	  --data_type="IID" \
      --server_address=$SERVER_ADDRESS 2>&1 | tee log/'edge2_'$curtime.txt &
    sleep 5
done
echo "Started $NUM_CLIENTS clients."
