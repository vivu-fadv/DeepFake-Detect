import os
import numpy as np
import pandas as pd
# TensorFlow and tf.keras
import tensorflow as tf
print('TensorFlow version: ', tf.__version__)



def configure_training_device():
    print('\n=== Device Check ===')
    print('Built with CUDA:', tf.test.is_built_with_cuda())
    print('Built with GPU support:', tf.test.is_built_with_gpu_support())
    build_info = tf.sysconfig.get_build_info()
    print('TensorFlow CUDA version:', build_info.get('cuda_version', 'unknown'))
    print('TensorFlow cuDNN version:', build_info.get('cudnn_version', 'unknown'))

    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')

    if gpus:
        print(f'Physical GPUs detected: {len(gpus)}')
        for index, gpu in enumerate(gpus):
            print(f'  GPU {index}: {gpu}')
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f'  Memory growth enabled for GPU {index}')
            except RuntimeError as exc:
                print(f'  Could not enable memory growth for GPU {index}: {exc}')

        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f'Logical GPUs available: {len(logical_gpus)}')
        for index, gpu in enumerate(logical_gpus):
            print(f'  Logical GPU {index}: {gpu}')
        print(f'CPUs available: {len(cpus)}')
        print('Training device selected: /GPU:0')
        print('GPU training enabled: YES')
        return '/GPU:0'

    print('Physical GPUs detected: 0')
    print('Logical GPUs available: 0')
    print(f'CPUs available: {len(cpus)}')
    print('Training device selected: /CPU:0')
    print('GPU training enabled: NO')
    print('WARNING: No NVIDIA GPU is visible to TensorFlow. Training will run on CPU.')
    return '/CPU:0'


TRAINING_DEVICE = configure_training_device()

dataset_path = './split_dataset/'

tmp_debug_path = './tmp_debug'
print('Creating Directory: ' + tmp_debug_path)
os.makedirs(tmp_debug_path, exist_ok=True)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# 224 is EfficientNetB0's native resolution — much better feature extraction than 128
input_size = 224
batch_size_num = 32
train_path = os.path.join(dataset_path, 'train')
val_path = os.path.join(dataset_path, 'val')
test_path = os.path.join(dataset_path, 'test')

# preprocess_input scales pixels to [-1, 1] which EfficientNet expects
# Stronger augmentation for deepfake detection
train_datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input,
    rotation_range = 15,
    horizontal_flip = True,
    zoom_range = 0.15,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    brightness_range = [0.8, 1.2],
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

# Compute class weights to handle imbalance
num_fake = np.sum(train_generator.classes == 0)
num_real = np.sum(train_generator.classes == 1)
total = num_fake + num_real
class_weight = {
    0: total / (2.0 * num_fake),
    1: total / (2.0 * num_real)
}
print(f'Class weights: {class_weight}')

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

with tf.device(TRAINING_DEVICE):
    # Build model with frozen base for Phase 1
    efficient_net = EfficientNetB0(
        weights = 'imagenet',
        input_shape = (input_size, input_size, 3),
        include_top = False,
        pooling = None  # We'll add our own pooling
    )

    # Freeze the base model for Phase 1
    efficient_net.trainable = False

    model = Sequential()
    model.add(efficient_net)
    model.add(GlobalAveragePooling2D())
    model.add(BatchNormalization())
    model.add(Dense(units = 256, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units = 1, activation = 'sigmoid'))
    model.summary()

checkpoint_filepath = '.\\tmp_checkpoint'
print('Creating Directory: ' + checkpoint_filepath)
os.makedirs(checkpoint_filepath, exist_ok=True)

# ============================================================
# Phase 1: Train head only (base frozen), higher learning rate
# ============================================================
print('\n=== Phase 1: Training head (base frozen) ===')
print('Phase 1 device:', TRAINING_DEVICE)
with tf.device(TRAINING_DEVICE):
    model.compile(
        optimizer = Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

phase1_callbacks = [
    EarlyStopping(
        monitor = 'val_loss',
        mode = 'min',
        patience = 5,
        verbose = 1,
        restore_best_weights = True
    ),
    ModelCheckpoint(
        filepath = os.path.join(checkpoint_filepath, 'best_model_phase1.keras'),
        monitor = 'val_loss',
        mode = 'min',
        verbose = 1,
        save_best_only = True
    ),
    ReduceLROnPlateau(
        monitor = 'val_loss',
        factor = 0.5,
        patience = 2,
        min_lr = 1e-5,
        verbose = 1
    )
]

with tf.device(TRAINING_DEVICE):
    history_phase1 = model.fit(
        train_generator,
        epochs = 15,
        steps_per_epoch = len(train_generator),
        validation_data = val_generator,
        validation_steps = len(val_generator),
        class_weight = class_weight,
        callbacks = phase1_callbacks
    )

# ============================================================
# Phase 2: Unfreeze all layers, fine-tune with very low lr
# ============================================================
print('\n=== Phase 2: Fine-tuning entire model ===')
efficient_net.trainable = True
print('Phase 2 device:', TRAINING_DEVICE)
with tf.device(TRAINING_DEVICE):
    model.compile(
        optimizer = Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

phase2_callbacks = [
    EarlyStopping(
        monitor = 'val_loss',
        mode = 'min',
        patience = 7,
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

with tf.device(TRAINING_DEVICE):
    history_phase2 = model.fit(
        train_generator,
        epochs = 30,
        steps_per_epoch = len(train_generator),
        validation_data = val_generator,
        validation_steps = len(val_generator),
        class_weight = class_weight,
        callbacks = phase2_callbacks
    )

# Load the best model from Phase 2
with tf.device(TRAINING_DEVICE):
    best_model = load_model(os.path.join(checkpoint_filepath, 'best_model.keras'))

# Also save a copy for the app
best_model.save('best_model.keras')

# Evaluate on test set
print('\n=== Evaluation on Test Set ===')
print('Evaluation device:', TRAINING_DEVICE)
test_generator.reset()
with tf.device(TRAINING_DEVICE):
    test_loss, test_accuracy = best_model.evaluate(test_generator, steps=len(test_generator), verbose=1)

# Generate predictions
test_generator.reset()
with tf.device(TRAINING_DEVICE):
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
