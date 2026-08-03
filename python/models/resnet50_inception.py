import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model

def create_model(input_shape=(224,224,3), num_classes=5):
    # Backbone: Pre-trained ResNet50
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # freeze for Stage 1 training

    x = base_model.output

    # Inception-style head
    branch1 = Conv2D(128, (1,1), activation='relu', padding='same')(x)
    branch3 = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    branch5 = Conv2D(128, (5,5), activation='relu', padding='same')(x)

    x = Concatenate()([branch1, branch3, branch5])
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model
