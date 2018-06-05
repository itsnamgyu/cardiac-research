import argparse
import os
import shutil

# Input format:
# INFO:tensorflow:                cap_augmented/OUT_OF_BASAL/DET0014101_SA3_ph0_r270.jpg  basal
# ...

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logfile', type=str, required=True)
    parser.add_argument('-d', '--dest_dir', type=str, required=True)
    args = parser.parse_args()

    labels = [
            'OUT_OF_BASAL',
            'OUT_OF_APICAL',
            'BASAL',
            'MIDDLE',
            'APICAL',
            ]
    # mind the order of OUT_OF_* and *

    with open (args.logfile) as log:
        for line in log:
            if not 'jpg' in line:
                continue

            for l in labels:
                if l in line:
                    truth = l
                    break
            else:
                continue

            values = line.split()
            path = values[1]
            prediction = '_'.join(values[2:]).upper()

            new_basename = truth + '_' + os.path.basename(path)
            save_dir = os.path.join(args.dest_dir, prediction)
            new_path = os.path.join(save_dir, new_basename)

            os.makedirs(save_dir, exist_ok=True)
            shutil.copy(path, new_path)

if __name__ == '__main__':
    main()
