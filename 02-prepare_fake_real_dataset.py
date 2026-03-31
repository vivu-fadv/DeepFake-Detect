import csv
import os
import shutil
import numpy as np
import splitfolders as split_folders

base_path = '.\\train_sample_videos\\Deepfakes\\'
dataset_path = '.\\prepared_dataset\\'
print('Creating Directory: ' + dataset_path)
os.makedirs(dataset_path, exist_ok=True)

tmp_fake_path = '.\\tmp_fake_faces'
print('Creating Directory: ' + tmp_fake_path)
os.makedirs(tmp_fake_path, exist_ok=True)

def get_filename_only(file_path):
    file_basename = os.path.basename(file_path)
    filename_only = file_basename.split('.')[0]
    return filename_only

with open(os.path.join('.\\train_sample_videos\\csv\\', 'Deepfakes.csv'), newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    metadata = {}
    for row in reader:
        metadata[row['File Path']] = row['Label'].strip().upper()
    print(len(metadata))

real_path = os.path.join(dataset_path, 'real')
print('Creating Directory: ' + real_path)
os.makedirs(real_path, exist_ok=True)

fake_path = os.path.join(dataset_path, 'fake')
print('Creating Directory: ' + fake_path)
os.makedirs(fake_path, exist_ok=True)

for filename, label in metadata.items():
    print(filename)
    print(label)
    tmp_path = os.path.join(os.path.join(base_path, get_filename_only(filename)), 'faces')
    print(tmp_path)
    if os.path.exists(tmp_path):
        if label == 'REAL':
            print('Copying to :' + real_path)
            shutil.copytree(tmp_path, real_path, dirs_exist_ok=True)
        elif label == 'FAKE':
            print('Copying to :' + tmp_fake_path)
            shutil.copytree(tmp_path, tmp_fake_path, dirs_exist_ok=True)
        else:
            print('Ignored..')

all_real_faces = [f for f in os.listdir(real_path) if os.path.isfile(os.path.join(real_path, f))]
print('Total Number of Real faces: ', len(all_real_faces))

all_fake_faces = [f for f in os.listdir(tmp_fake_path) if os.path.isfile(os.path.join(tmp_fake_path, f))]
print('Total Number of Fake faces: ', len(all_fake_faces))

random_faces = np.random.choice(all_fake_faces, len(all_real_faces), replace=False)
for fname in random_faces:
    src = os.path.join(tmp_fake_path, fname)
    dst = os.path.join(fake_path, fname)
    shutil.copyfile(src, dst)

print('Down-sampling Done!')

# Split into Train/ Val/ Test folders
split_folders.ratio(dataset_path, output='split_dataset', seed=1377, ratio=(.8, .1, .1)) # default values
print('Train/ Val/ Test Split Done!')