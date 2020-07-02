# Cardiac Research

Automatic classification of short axis cardiac CINE MRI images. A comparison between
the feature extraction setting and layerwise fine-tuning.

_Accepted for ISMRM 27th (2019) poster in Montreal! Journal submission pending. 💃🕺_

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

## Run Latest Feature Extraction Setting Experiment

```
cd experiments/deep_a0
cp params.example.py params.py
python3 main.py
```
