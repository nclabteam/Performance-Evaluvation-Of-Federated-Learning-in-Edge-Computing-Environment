#!/bin/bash
kubectl apply -f cloud.yaml
sleep 10
kubectl apply -f client.yaml
echo "Start training model..."
