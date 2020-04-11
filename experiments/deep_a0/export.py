import os
import glob
import shutil

input_dir = os.path.abspath('output')
output_dir = os.path.abspath('output_export')
shutil.rmtree(output_dir, ignore_errors=True)
os.makedirs(output_dir, exist_ok=True)

ls = list(glob.glob('output/**/*'))
ls_path = os.path.join(output_dir, 'ls.txt')
with open(ls_path, 'w') as f:
    f.write('\n'.join(ls))

final_dirs = glob.glob(os.path.join(input_dir, '*/D00_FINAL'))
for final_dir in final_dirs:
    target = final_dir.replace(input_dir, output_dir)
    shutil.copytree(final_dir, target)

history_paths = glob.glob(
    os.path.join(input_dir, '*/D00_L*/train_val_history.*'))
for history_path in history_paths:
    target = history_path.replace(input_dir, output_dir)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    shutil.copy(history_path, target)

print('Output files selected and copied to {}'.format(output_dir))
