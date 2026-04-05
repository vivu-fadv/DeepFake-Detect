import json
import os
from distutils.dir_util import copy_tree
import shutil
import pandas as pd

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras import backend as K
print('TensorFlow version: ', tf.__version__)

# Set to force CPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#if tf.test.gpu_device_name():
#    print('GPU found')
#else:
#    print("No GPU found")

dataset_path = '.\\split_dataset\\'

tmp_debug_path = '.\\tmp_debug'
print('Creating Directory: ' + tmp_debug_path)
os.makedirs(tmp_debug_path, exist_ok=True)

def get_filename_only(file_path):
    file_basename = os.path.basename(file_path)
    filename_only = file_basename.split('.')[0]
    return filename_only

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import applications
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

input_size = 128
batch_size_num = 32
train_path = os.path.join(dataset_path, 'train')
val_path = os.path.join(dataset_path, 'val')
test_path = os.path.join(dataset_path, 'test')

train_datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input,
    rotation_range = 15,
    width_shift_range = 0.15,
    height_shift_range = 0.15,
    shear_range = 0.2,
    zoom_range = 0.15,
    horizontal_flip = True,
    brightness_range = [0.8, 1.2],
    channel_shift_range = 30,
    fill_mode = 'nearest'
)

train_generator = train_datagen.flow_from_directory(
    directory = train_path,
    target_size = (input_size, input_size),
    color_mode = "rgb",
    class_mode = "binary",
    batch_size = batch_size_num,
    shuffle = True
)

# Compute class weights to handle imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)
class_weight_dict = dict(enumerate(class_weights))
print(f'Class mapping: {train_generator.class_indices}')
print(f'Class weights: {class_weight_dict}')
print(f'Train samples - fake: {np.sum(train_generator.classes == 0)}, real: {np.sum(train_generator.classes == 1)}')

val_datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input
)

val_generator = val_datagen.flow_from_directory(
    directory = val_path,
    target_size = (input_size, input_size),
    color_mode = "rgb",
    class_mode = "binary",
    batch_size = batch_size_num,
    shuffle = True
)

test_datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input
)

test_generator = test_datagen.flow_from_directory(
    directory = test_path,
    classes=['fake', 'real'],
    target_size = (input_size, input_size),
    color_mode = "rgb",
    class_mode = "binary",
    batch_size = 1,
    shuffle = False
)

# --- Phase 1: Train with frozen base ---
efficient_net = EfficientNetB0(
    weights = 'imagenet',
    input_shape = (input_size, input_size, 3),
    include_top = False,
    pooling = 'max'
)
efficient_net.trainable = False  # freeze base initially

model = Sequential()
model.add(efficient_net)
model.add(Dense(units = 512, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(units = 1, activation = 'sigmoid'))
model.summary()

model.compile(optimizer = Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

checkpoint_filepath = '.\\tmp_checkpoint'
print('Creating Directory: ' + checkpoint_filepath)
os.makedirs(checkpoint_filepath, exist_ok=True)

custom_callbacks = [
    EarlyStopping(
        monitor = 'val_accuracy',
        mode = 'max',
        patience = 5,
        verbose = 1,
        restore_best_weights = True
    ),
    ModelCheckpoint(
        filepath = os.path.join(checkpoint_filepath, 'best_model.keras'),
        monitor = 'val_accuracy',
        mode = 'max',
        verbose = 1,
        save_best_only = True
    ),
    ReduceLROnPlateau(
        monitor = 'val_accuracy',
        factor = 0.5,
        patience = 3,
        min_lr = 1e-7,
        verbose = 1,
        mode = 'max'
    )
]

print('\n=== Phase 1: Training with frozen base ===')
num_epochs = 15
history = model.fit(
    train_generator,
    epochs = num_epochs,
    steps_per_epoch = len(train_generator),
    validation_data = val_generator,
    validation_steps = len(val_generator),
    callbacks = custom_callbacks,
    class_weight = class_weight_dict
)

# --- Phase 2: Fine-tune top layers of base model ---
print('\n=== Phase 2: Fine-tuning top layers ===')
efficient_net.trainable = True
# Freeze all layers except the last 30
for layer in efficient_net.layers[:-30]:
    layer.trainable = False

model.compile(optimizer = Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

fine_tune_callbacks = [
    EarlyStopping(
        monitor = 'val_accuracy',
        mode = 'max',
        patience = 5,
        verbose = 1,
        restore_best_weights = True
    ),
    ModelCheckpoint(
        filepath = os.path.join(checkpoint_filepath, 'best_model.keras'),
        monitor = 'val_accuracy',
        mode = 'max',
        verbose = 1,
        save_best_only = True
    ),
    ReduceLROnPlateau(
        monitor = 'val_accuracy',
        factor = 0.5,
        patience = 3,
        min_lr = 1e-8,
        verbose = 1,
        mode = 'max'
    )
]

fine_tune_epochs = 30
history_fine = model.fit(
    train_generator,
    epochs = fine_tune_epochs,
    steps_per_epoch = len(train_generator),
    validation_data = val_generator,
    validation_steps = len(val_generator),
    callbacks = fine_tune_callbacks,
    class_weight = class_weight_dict
)

# Load the best model
best_model = load_model(os.path.join(checkpoint_filepath, 'best_model.keras'))

# Evaluate on test set
print('\n=== Evaluation on Test Set ===')
test_generator.reset()
test_loss, test_accuracy = best_model.evaluate(test_generator, steps=len(test_generator), verbose=1)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# Generate predictions
test_generator.reset()
preds = best_model.predict(test_generator, verbose=1)
pred_labels = (preds.flatten() > 0.5).astype(int)
true_labels = test_generator.classes

from sklearn.metrics import classification_report, confusion_matrix
print('\nClassification Report:')
print(classification_report(true_labels, pred_labels, target_names=['fake', 'real']))
print('Confusion Matrix:')
print(confusion_matrix(true_labels, pred_labels))

test_results = pd.DataFrame({
    "Filename": test_generator.filenames,
    "Prediction": preds.flatten(),
    "Predicted_Label": pred_labels,
    "True_Label": true_labels
})
print(test_results)