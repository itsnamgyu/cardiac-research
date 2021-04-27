# Cardiac Research

Official code for Ho, N., Kim, YC. Evaluation of transfer learning in deep convolutional neural network models for cardiac short axis slice classification. Sci Rep 11, 1839 (2021). https://doi.org/10.1038/s41598-021-81525-9

Major clean-up pending.

Automatic classification of short axis cardiac CINE MRI images. A comparison between
the feature extraction setting and layerwise fine-tuning.

## Environment

- python>=3.6
- tensorflow 1.11+ (tested on 1.11, 1.15)
- keras~=2.2.4

## Setup

Virtual environment recommended. Will not add instructions on venv.

```sh
pip install -r requirements.txt
python3 setup.py develop
```

## Setup for conda (recommended for GPU)

```
conda create -n envname python=3.7
conda install tensorflow-gpu==1.6  # -gpu optional
conda activate envname
pip install -r requirements.txt
python3 setup.py develop
```

## Run Latest Feature Extraction Setting Experiment

```
cd experiments/deep_a0
cp params.example.py params.py
python3 main.py
```

