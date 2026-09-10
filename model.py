"""
Stage 5: CNN Architecture Module
--------------------------------
Defines a lightweight, CPU-efficient Convolutional Neural Network (CNN)
using modern TensorFlow/Keras APIs.

Architecture:
- Input: (28, 28, 1)
- ConvBlock 1: Conv2D(32, 3x3) -> BatchNormalization -> ReLU -> MaxPooling2D(2x2)
- ConvBlock 2: Conv2D(64, 3x3) -> BatchNormalization -> ReLU -> MaxPooling2D(2x2)
- ConvBlock 3: Conv2D(128, 3x3) -> ReLU -> Dropout(0.25)
- Classifier: Flatten -> Dense(128, ReLU) -> Dropout(0.4) -> Dense(num_classes=36, Softmax)
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_character_cnn(input_shape=(28, 28, 1), num_classes=36, learning_rate=0.001):
    """
    Builds and compiles a lightweight, modern CNN model for handwritten character recognition.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of single image, default (28, 28, 1).
    num_classes : int
        Number of output categories, default 36 (digits 0-9 + alphabets A-Z).
    learning_rate : float
        Initial learning rate for Adam optimizer.
        
    Returns:
    --------
    model : tf.keras.Model
        Compiled Keras model.
    """
    model = keras.Sequential([
        # First Convolutional Block
        layers.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', name='conv1'),
        layers.BatchNormalization(name='bn1'),
        layers.MaxPooling2D(pool_size=(2, 2), name='pool1'),
        
        # Second Convolutional Block
        layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', name='conv2'),
        layers.BatchNormalization(name='bn2'),
        layers.MaxPooling2D(pool_size=(2, 2), name='pool2'),
        
        # Third Convolutional Block
        layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu', name='conv3'),
        layers.Dropout(0.25, name='drop1'),
        
        # Fully Connected Classifier
        layers.Flatten(name='flatten'),
        layers.Dense(128, activation='relu', name='dense1'),
        layers.Dropout(0.4, name='drop2'),
        layers.Dense(num_classes, activation='softmax', name='output')
    ], name='HandwrittenCharacterCNN')
    
    # Compile model with Adam and Sparse Categorical Crossentropy
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    print("Testing CNN architecture...")
    model = build_character_cnn()
    model.summary()
    print("CNN architecture built and verified successfully!")
