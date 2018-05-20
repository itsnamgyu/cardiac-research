## Data Preparation
Unzip CAP Challenge Training Set.tar to folder *cap_challenge* in the current folder.

### Example File Path
```
labeler/cap_challenge/DET0000101/DET0000101_SA1_ph0.dcm 
```

## Install Requirements
```
pip install -r requirements.txt
```

## Run Module
```
python3 labeler.py
```

## Usage
1. Use the number keys 0-5 to select the labeling mode. The labeling modes are as follows:
```
0: UNLABELED
1: OUT_OF_APICAL
2: APICAL
3: MIDDLE
4: BASAL
5: OUT_OF_BASAL
```
2. Left click on the MRI images to assign the selected label
3. Use the left, right arrow keys to move between subjects
4. Use the h, l keys to quickly skip between subjects (10 at a time)
5. Use the t key to switch between color modes
5. Close the plot window and the labeling data will be saved to `label_data.npy`, and the labeled images will be saved to directory `cap_labeled`

## Test First!
*I recommend to test the data save/load functions before you label  multiple images.*