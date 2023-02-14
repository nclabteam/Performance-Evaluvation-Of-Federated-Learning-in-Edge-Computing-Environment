#!/bin/bash
kubectl delete -f cloud.yaml
kubectl delete -f client.yaml
echo "Deleted training model!!!"
