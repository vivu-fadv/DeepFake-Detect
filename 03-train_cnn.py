import os
import pandas as pd
import numpy as np

# TensorFlow and tf.keras
import tensorflow as tf
print('TensorFlow version: ', tf.__version__)

dataset_path = '.\\split_dataset\\'

tmp_debug_path = '.\\tmp_debug'
print('Creating Directory: ' + tmp_debug_path)
os.makedirs(tmp_debug_path, exist_ok=True)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

# preprocess_input scales pixels to [-1, 1] which EfficientNet expects
train_datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input,
    rotation_range = 10,
    horizontal_flip = True,
    zoom_range = 0.1,
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

print(f'Class mapping: {train_generator.class_indices}')
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

# Build model - entire EfficientNetB0 is trainable
efficient_net = EfficientNetB0(
    weights = 'imagenet',
    input_shape = (input_size, input_size, 3),
    include_top = False,
    pooling = 'max'
)

model = Sequential()
model.add(efficient_net)
model.add(Dense(units = 512, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))
model.summary()

model.compile(optimizer = Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

checkpoint_filepath = '.\\tmp_checkpoint'
print('Creating Directory: ' + checkpoint_filepath)
os.makedirs(checkpoint_filepath, exist_ok=True)

callbacks = [
    EarlyStopping(
        monitor = 'val_loss',
        mode = 'min',
        patience = 5,
        verbose = 1,
        restore_best_weights = True
    ),
    ModelCheckpoint(
        filepath = os.path.join(checkpoint_filepath, 'best_model.keras'),
        monitor = 'val_loss',
        mode = 'min',
        verbose = 1,
        save_best_only = True
    ),
    ReduceLROnPlateau(
        monitor = 'val_loss',
        factor = 0.5,
        patience = 3,
        min_lr = 1e-7,
        verbose = 1
    )
]

print('\n=== Training ===')
num_epochs = 20
history = model.fit(
    train_generator,
    epochs = num_epochs,
    steps_per_epoch = len(train_generator),
    validation_data = val_generator,
    validation_steps = len(val_generator),
    callbacks = callbacks
)

# Load the best model
best_model = load_model(os.path.join(checkpoint_filepath, 'best_model.keras'))

# Evaluate on test set
print('\n=== Evaluation on Test Set ===')
test_generator.reset()
test_loss, test_accuracy = best_model.evaluate(test_generator, steps=len(test_generator), verbose=1)

# Generate predictions
test_generator.reset()
preds = best_model.predict(test_generator, verbose=1)
pred_labels = (preds.flatten() > 0.5).astype(int)
true_labels = test_generator.classes

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

overall_accuracy = accuracy_score(true_labels, pred_labels)
cm = confusion_matrix(true_labels, pred_labels)

print(f'\n{"="*60}')
print(f'  MODEL ACCURACY REPORT')
print(f'{"="*60}')
print(f'  Overall Accuracy:  {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)')
print(f'  Test Loss:         {test_loss:.4f}')
print(f'{"="*60}')

# Per-class accuracy
fake_correct = cm[0][0]
fake_total = cm[0].sum()
real_correct = cm[1][1]
real_total = cm[1].sum()
print(f'  Fake  Accuracy:    {fake_correct}/{fake_total} = {fake_correct/fake_total:.4f} ({fake_correct/fake_total*100:.2f}%)')
print(f'  Real  Accuracy:    {real_correct}/{real_total} = {real_correct/real_total:.4f} ({real_correct/real_total*100:.2f}%)')
print(f'{"="*60}')

print('\nClassification Report:')
print(classification_report(true_labels, pred_labels, target_names=['fake', 'real']))
print('Confusion Matrix:')
print(cm)

test_results = pd.DataFrame({
    "Filename": test_generator.filenames,
    "Prediction": preds.flatten(),
    "Predicted_Label": pred_labels,
    "True_Label": true_labels
})
print(test_results)