# 3D Mesh Saliency Prediction

This repository contains the code for training deep learning models for predicting saliency on 3D meshes.

## Dataset

You can find the dataset in the `3d-meshes` folder.

## Training Code

You can find the training code for each model in these folders:

-    original-model
-    optimized-model
-    new-model

## Trained Models

-    saved-models/original-model-16.h5
-    saved-models/original-model-32.h5
-    saved-models/original-model-64.h5
-    saved-models/optimized-model-16.h5
-    saved-models/optimized-model-32.h5
-    saved-models/optimized-model-64.h5
-    saved-models/new-model-16.h5
-    saved-models/new-model-32.h5
-    saved-models/new-model-64.h5

## Infrence

to run the infrence code, run the following command:

```bash
python predict.py --patchSize=16 --model_name=saved-models/original-model-16.h5 --mesh_path=3d-meshes/vase_decimated.obj
```

## Cache Patches

To cache the patches, run the following command:

```bash
python3 generate_patches.py --patchSize=16
```

To see a comparison between all the models, run the notebook `visualization.ipynb`. (caching the patches is required)

## Requirements

-    tensorflow
-    numpy
-    trimesh
-    sklearn
