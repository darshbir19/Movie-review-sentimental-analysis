import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout


def get_cnn_model(max_len):
    """
    Build and compile a 1D Convolutional Neural Network (CNN) model for
    binary text classification.

    The model uses an embedding layer to learn word representations,
    followed by a 1D convolution and global max pooling to capture
    local n-gram features from text sequences.

    Architecture:
    - Input layer of length `max_len` (padded token sequences)
    - Embedding layer with:
        * Vocabulary size: 601
        * Embedding dimension: 100
    - 1D Convolutional layer with:
        * 64 filters
        * Kernel size of 3
        * ReLU activation
    - Global Max Pooling layer
    - Dense layer with 32 units and ReLU activation
    - Dropout layer with rate 0.4
    - Output Dense layer with 1 unit and sigmoid activation

    Compilation:
    - Optimizer: Adam
    - Loss function: Binary Crossentropy
    - Metric: Accuracy

    Parameters
    ----------
    max_len : int
        Maximum length of input sequences after padding/truncation.

    Returns
    -------
    tensorflow.keras.models.Sequential
        A compiled Keras CNN model ready for training.
    """

    cnn_model = Sequential([
        Input(shape=(max_len,)),
        Embedding(input_dim=601, output_dim=100),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(32, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])

    cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return cnn_model
