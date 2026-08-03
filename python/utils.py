import matplotlib.pyplot as plt
import numpy as np

def plot_training(history1, history2):
    plt.figure(figsize=(12,5))
    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(history1.history['accuracy'], label='stage1_train')
    plt.plot(history1.history['val_accuracy'], label='stage1_val')
    plt.plot(history2.history['accuracy'], label='stage2_train')
    plt.plot(history2.history['val_accuracy'], label='stage2_val')
    plt.legend()
    plt.title('Accuracy')

    # Loss
    plt.subplot(1,2,2)
    plt.plot(history1.history['loss'], label='stage1_train')
    plt.plot(history1.history['val_loss'], label='stage1_val')
    plt.plot(history2.history['loss'], label='stage2_train')
    plt.plot(history2.history['val_loss'], label='stage2_val')
    plt.legend()
    plt.title('Loss')
    plt.show()

def compute_class_weights(labels):
    """labels: list/array of class indices"""
    classes, counts = np.unique(labels, return_counts=True)
    total = sum(counts)
    weights = {cls: total/count for cls, count in zip(classes, counts)}
    return weights
