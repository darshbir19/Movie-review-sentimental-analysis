import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout


def get_mlp_model():
  """
    Build and compile a Multilayer Perceptron (MLP) model for binary classification.

    The model consists of fully connected (Dense) layers with ReLU activation,
    Dropout regularization to reduce overfitting, and a sigmoid-activated output
    neuron for binary classification tasks.

    Architecture:
    - Input layer of size 5000 (e.g., TF-IDF feature vector)
    - Dense layer with 32 units and ReLU activation
    - Dropout layer with rate 0.6
    - Dense layer with 16 units and ReLU activation
    - Dropout layer with rate 0.6
    - Output Dense layer with 1 unit and sigmoid activation

    Compilation:
    - Optimizer: Adam
    - Loss function: Binary Crossentropy
    - Metric: Accuracy

    Returns
    -------
    tensorflow.keras.models.Sequential
        A compiled Keras Sequential MLP model ready for training.
    """
  mlp_model = Sequential([
        Input(shape=(5000,)),
        Dense(32, activation='relu'),
        Dropout(0.6),
        Dense(16, activation='relu'),
        Dropout(0.6),
        Dense(1, activation='sigmoid')
        ])

  mlp_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

  return mlp_model
