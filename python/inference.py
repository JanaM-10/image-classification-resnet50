import numpy as np
from PIL import Image
import tensorflow as tf
from models.resnet50_inception import create_model

model_path = 'results/models/final_model.h5'
patch_size = 224
threshold_secondary = 0.05

# Load model
model = tf.keras.models.load_model(model_path)

def patch_based_predict(img_path):
    img = Image.open(img_path).convert('RGB')
    img_width, img_height = img.size
    class_counts = np.zeros((5,))  # 5 classes

    # Sliding window
    for top in range(0, img_height, patch_size):
        for left in range(0, img_width, patch_size):
            patch = img.crop((left, top, left+patch_size, top+patch_size)).resize((224,224))
            x = np.expand_dims(np.array(patch)/255.0, axis=0)
            preds = model.predict(x)
            class_counts += preds[0]

    dominant_class = np.argmax(class_counts)
    total = np.sum(class_counts)
    secondary_classes = [i for i, c in enumerate(class_counts/total) if c > threshold_secondary and i != dominant_class]

    return dominant_class, secondary_classes

# Example
dominant, secondary = patch_based_predict('dataset/test/sample.jpg')
print(f'Dominant class: {dominant}, Secondary classes: {secondary}')
