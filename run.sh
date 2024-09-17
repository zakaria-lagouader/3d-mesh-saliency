#!/bin/bash

for file in objects/*.obj; do
    python predict.py --patchSize=64 --model_name=saved-models/new-model-64.h5 --mesh_path="$file" --export_folder=exports
done

# find objects -name "*.obj" | parallel python predict.py --patchSize=64 --model_name=saved-models/new-model-64.h5 --mesh_path={} --export_folder=exports