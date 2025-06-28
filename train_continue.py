import os
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Paths
train_dir = 'files/train'
test_dir = 'files/test'
model_path = '_model/skin_model.h5'
labels_path = '_model/labels.json'

# Image properties
img_width, img_height = 224, 224
batch_size = 32
epochs = 20  # Additional epochs

# Load model
if os.path.exists(model_path):
    model = load_model(model_path)
    print("✅ Loaded existing model.")

    # 🔁 Re-compile with lower learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
else:
    print("❌ Model not found. Please train from scratch first.")
    exit()

# Load label map
with open(labels_path, 'r') as f:
    class_indices = json.load(f)
num_classes = len(class_indices)

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)

# Simple normalization for validation
test_datagen = ImageDataGenerator(rescale=1./255)

# Data loaders
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# Callbacks
checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# Continue training
model.fit(
    train_data,
    validation_data=test_data,
    epochs=epochs,
    callbacks=[checkpoint, early_stop]
)
