import csv
import os
import shutil
import numpy as np
import splitfolders as split_folders
from PIL import Image

MIN_IMAGE_SIZE = 90  # minimum width and height in pixels

base_path = '.\\train_sample_videos\\Deepfakes\\'
dataset_path = '.\\prepared_dataset\\'
print('Creating Directory: ' + dataset_path)
os.makedirs(dataset_path, exist_ok=True)

tmp_fake_path = '.\\tmp_fake_faces'
print('Creating Directory: ' + tmp_fake_path)
if os.path.exists(tmp_fake_path):
    shutil.rmtree(tmp_fake_path)
os.makedirs(tmp_fake_path)

def get_filename_only(file_path):
    file_basename = os.path.basename(file_path)
    filename_only = file_basename.split('.')[0]
    return filename_only

def copy_large_faces(src_dir, dst_dir):
    """Copy only images that are at least MIN_IMAGE_SIZE x MIN_IMAGE_SIZE."""
    skipped = 0
    copied = 0
    for fname in os.listdir(src_dir):
        src_file = os.path.join(src_dir, fname)
        if not os.path.isfile(src_file):
            continue
        try:
            with Image.open(src_file) as img:
                w, h = img.size
            if w >= MIN_IMAGE_SIZE and h >= MIN_IMAGE_SIZE:
                shutil.copy2(src_file, os.path.join(dst_dir, fname))
                copied += 1
                print(f'Copied: {copied}')
            else:
                print(f'Skipped {fname}: {w}x{h}')
                skipped += 1
        except Exception:
            skipped += 1
   
with open(os.path.join('.\\train_sample_videos\\csv\\', 'Deepfakes.csv'), newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    metadata = {}
    for row in reader:
        metadata[row['File Path']] = row['Label'].strip().upper()
    print(len(metadata))

real_path = os.path.join(dataset_path, 'real')
print('Creating Directory: ' + real_path)
if os.path.exists(real_path):
    shutil.rmtree(real_path)
os.makedirs(real_path)

fake_path = os.path.join(dataset_path, 'fake')
print('Creating Directory: ' + fake_path)
if os.path.exists(fake_path):
    shutil.rmtree(fake_path)
os.makedirs(fake_path)

for filename, label in metadata.items():
    print(filename)
    print(label)
    tmp_path = os.path.join(os.path.join(base_path, get_filename_only(filename)), 'faces')
    print(tmp_path)
    if os.path.exists(tmp_path):
        if label == 'REAL':
            print('Copying to :' + real_path)
            copy_large_faces(tmp_path, real_path)
        elif label == 'FAKE':
            print('Copying to :' + tmp_fake_path)
            copy_large_faces(tmp_path, tmp_fake_path)
        else:
            print('Ignored..')

all_real_faces = [f for f in os.listdir(real_path) if os.path.isfile(os.path.join(real_path, f))]
print('Total Number of Real faces: ', len(all_real_faces))

all_fake_faces = [f for f in os.listdir(tmp_fake_path) if os.path.isfile(os.path.join(tmp_fake_path, f))]
print('Total Number of Fake faces: ', len(all_fake_faces))

print('Copying filtered fake faces to: ' + fake_path)
copy_large_faces(tmp_fake_path, fake_path)

print('Copying all fake faces Done!')

# Split into Train/ Val/ Test folders
split_folders.ratio(dataset_path, output='split_dataset', seed=1377, ratio=(.8, .1, .1)) # default values
print('Train/ Val/ Test Split Done!')