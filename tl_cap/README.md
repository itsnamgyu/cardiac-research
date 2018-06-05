# TensorFlow Retraining Module for CAP Images

This module automatically trains, validates, and tests retrained classifiers and outputs misclassified test images into a seperate directory for visual examination.

## Usage
1. Place augmented images in subdirectory `cap_augmented`
2. Use the run.sh convenience script with the following format.
```bash
source run.sh 0.001 100000 mobile_0001 https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/1
```
`0.001`: learning rate

`100000`: traing steps

`mobile_0001`: name of subdirectory for output files (stated below)

`https://...mobilenet_v2...`: tfhub module url of original image recognition model

Refer to run_example.sh


## Output Files (By Directory)
`train_misclassified`: misclassified images
`train_output`: stdout of retrain.py
`train_models`: trained models
`train_summary`: TF summaries for use with TensorBoard


## Advanced Usage
```
python3 retrain.py -h
```

## TF-Hub Modules
Refer to `tfhub_modules.txt` for common tfhub modules
