import pandas as pd
import matplotlib.pyplot as plt
import nltk
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from preprocess import (
    remove_duplicates,
    clean_text,
    text_preprocessing_mlp,
    tfidf_vectorization,
    split_dataset,
    tokenize_text
)

from mlp_model import get_mlp_model
from cnn_model import get_cnn_model

# ----------------------------
# Load and clean dataset
# ----------------------------
movies_data = pd.read_csv(r"src/movies_data.csv")
movies_data = remove_duplicates(movies_data)
movies_data["cleaned_text"] = movies_data.text.apply(clean_text)

# ----------------------------
# Download NLTK resources
# ----------------------------
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# ----------------------------
# Preprocess text for MLP
# ----------------------------
movies_data["preprocessed_text_mlp"] = movies_data.cleaned_text.apply(
    text_preprocessing_mlp
)

# TF-IDF vectorization
X_tfidf, tfidf_vectorizer = tfidf_vectorization(
    movies_data["preprocessed_text_mlp"], max_features=5000
)

# Split dataset
X_train, X_test, y_train, y_test = split_dataset(X_tfidf, movies_data)

# ----------------------------
# Train MLP model
# ----------------------------
mlp_model = get_mlp_model()
mlp_model.summary()

history = mlp_model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2

)

loss, accuracy = mlp_model.evaluate(X_test, y_test)
print(f"MLP Loss: {loss:.3f}, Accuracy: {accuracy:.3f}")

# Confusion matrix
predictions = mlp_model.predict(X_test)
pred_labels = (predictions > 0.5).astype(int)
cm = confusion_matrix(y_test, pred_labels)
ConfusionMatrixDisplay(cm, display_labels=[0, 1]).plot(cmap="BuGn")
plt.title("MLP Confusion Matrix")
plt.show()

mlp_model.save("mlp_model.keras")

# ----------------------------
# Preprocess text for CNN
# ----------------------------
padded_sequences, tokenizer = tokenize_text(movies_data["cleaned_text"])

X_train, X_test, y_train, y_test = split_dataset(padded_sequences, movies_data)

# ----------------------------
# Train CNN model
# ----------------------------
max_len = 200
cnn_model = get_cnn_model(max_len)
cnn_model.summary()

history = cnn_model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2
)

loss, accuracy = cnn_model.evaluate(X_test, y_test)
print(f"CNN Loss: {loss:.3f}, Accuracy: {accuracy:.3f}")

# Confusion matrix
predictions = cnn_model.predict(X_test)
pred_labels = (predictions > 0.5).astype(int)
cm = confusion_matrix(y_test, pred_labels)
ConfusionMatrixDisplay(cm, display_labels=[0, 1]).plot(cmap="BuGn")
plt.title("CNN Confusion Matrix")
plt.show()

cnn_model.save("cnn_model.keras")
