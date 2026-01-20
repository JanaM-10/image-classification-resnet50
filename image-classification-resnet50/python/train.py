import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models.resnet50_inception import create_model

# Paths
train_dir = 'dataset/train'
val_dir = 'dataset/val'
model_save_path = 'results/models/final_model.h5'

# Data augmentation
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True
)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(train_dir, target_size=(224,224), batch_size=32, class_mode='categorical')
val_data = val_gen.flow_from_directory(val_dir, target_size=(224,224), batch_size=32, class_mode='categorical')

# Create model
model = create_model()

# Compile for Stage 1
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Stage 1: Train head
history_stage1 = model.fit(train_data, validation_data=val_data, epochs=15)

# Stage 2: Fine-tune last 30 layers
for layer in model.layers[-30:]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_stage2 = model.fit(train_data, validation_data=val_data, epochs=10)

# Save model
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
model.save(model_save_path)
print(f'Model saved at {model_save_path}')
