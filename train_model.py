# train_model.py (MobileNetV2 version)
import os
import json
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

# Path
train_dir = 'files/train'
test_dir = 'files/test'
model_dir = '_model'
os.makedirs(model_dir, exist_ok=True)

# Image property
img_width, img_height = 224, 224
batch_size = 32
epochs = 20

# Data generator
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

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

num_classes = len(train_data.class_indices)

# class labels
with open(os.path.join(model_dir, 'labels.json'), 'w') as f:
    json.dump(train_data.class_indices, f)

# Load  base
base_model = MobileNetV2(include_top=False, input_shape=(img_width, img_height, 3), weights='imagenet')
base_model.trainable = False  # freeze base

#  custom layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callback
checkpoint_path = os.path.join(model_dir, 'skin_model.h5')
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True)
early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

# Training model
model.fit(
    train_data,
    validation_data=test_data,
    epochs=epochs,
    callbacks=[checkpoint, early_stop]
)
