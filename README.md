# Cardiac Research

Automatic classification of short axis cardiac CINE MRI images.

## Module Explanation
TODO

## Environment Setup
1. Install Keras 2.2.4 and your TensorFlow build of choice.
2. Install required Python packages in `requirements.txt`.
3. Run the setup script in development mode.
```
python3 setup.py develop
```
## Run Latest Deep Tuning Optimization Script
```
cd experiments/deep_1
python3 run_single.py
```

Fix line 13 in run_single.py to change the base network
```
fm = models['mobileneta25']()  # change key 'mobileneta25'
