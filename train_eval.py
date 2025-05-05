# train_eval.py
import numpy as np
from data import train_data, test_data
from preprocess import build_vocab, preprocess_data, text_to_sequence # Varsayılan preprocess modülü
from rnn_from_scratch import RNNFromScratch # Sıfırdan RNN
from rnn_library import create_library_rnn_model # Kütüphane RNN'i
from tensorflow.keras.preprocessing.sequence import pad_sequences # Kütüphane modeli için padding
# Grafik çizimi ve metrikler için (isteğe bağlı ama raporda gerekli)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# --- Veri Hazırlığı ---
print("1. Preparing Data...")
# Tüm kelimeleri içeren sözlüğü oluştur
vocab = build_vocab({**train_data, **test_data})
vocab_size = len(vocab)
print(f"Vocabulary Size: {vocab_size}")

# Eğitim ve test verisini işle
X_train_seq, y_train = preprocess_data(train_data, vocab)
X_test_seq, y_test = preprocess_data(test_data, vocab)

# Padding için maksimum dizi uzunluğunu bul (Kütüphane modeli için gerekli)
all_seqs = X_train_seq + X_test_seq
max_length = max(len(x) for x in all_seqs) if all_seqs else 0
print(f"Max sequence length: {max_length}")

X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')


# --- Model 1: Sıfırdan RNN ---
print("\n2. RNN From Scratch...")
# Not: RNNFromScratch sınıfının train ve predict metodları tam implementasyon gerektirir.
rnn_scratch = RNNFromScratch(vocab_size=vocab_size, hidden_size=32, output_size=1)
# rnn_scratch.train(X_train_seq, y_train, learning_rate=0.01, epochs=5) # Eğitim (implementasyon gerekli)
print("Skipping training for RNN From Scratch (requires full implementation).")

# Değerlendirme (implementasyon sonrası)
# y_pred_scratch = np.array([rnn_scratch.predict(seq) for seq in X_test_seq])
# accuracy_scratch = accuracy_score(y_test, y_pred_scratch)
# print(f"RNN From Scratch - Test Accuracy: {accuracy_scratch:.4f} (Placeholder)")
accuracy_scratch = 0.0 # Placeholder
y_pred_scratch = np.random.randint(0, 2, size=len(y_test)) # Placeholder predictions

# --- Model 2: Kütüphane RNN ---
print("\n3. Library RNN (TensorFlow/Keras)...")
rnn_lib = create_library_rnn_model(vocab_size=vocab_size, embedding_dim=16, rnn_units=32, max_length=max_length)
rnn_lib.summary()

print("Training Library RNN Model...")
history = rnn_lib.fit(X_train_pad, y_train, epochs=15, validation_split=0.2, batch_size=8, verbose=1)

print("Evaluating Library RNN Model...")
loss_lib, accuracy_lib = rnn_lib.evaluate(X_test_pad, y_test, verbose=0)
print(f"Library RNN - Test Loss: {loss_lib:.4f}")
print(f"Library RNN - Test Accuracy: {accuracy_lib:.4f}")

# Tahminler (Metrikler için)
y_pred_prob_lib = rnn_lib.predict(X_test_pad)
y_pred_lib = (y_pred_prob_lib > 0.5).astype(int).flatten()


# --- Sonuçları Raporlama ve Görselleştirme [cite: 9] ---
print("\n4. Results & Visualization...")

# Doğruluk Karşılaştırması
print(f"\nAccuracy Comparison:")
print(f" - RNN From Scratch: {accuracy_scratch:.4f} (Requires implementation)")
print(f" - Library RNN:      {accuracy_lib:.4f}")

# Kütüphane Modeli için Eğitim Grafikleri
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Library Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Library Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("library_model_training_curves.png") # Grafiği kaydet
print("Saved training curves graph to 'library_model_training_curves.png'")
# plt.show() # İsteğe bağlı olarak grafikleri göster

# Karmaşıklık Matrisleri [cite: 9]
print("\nConfusion Matrices:")

# Sıfırdan RNN (Implementasyon sonrası)
# cm_scratch = confusion_matrix(y_test, y_pred_scratch)
# disp_scratch = ConfusionMatrixDisplay(confusion_matrix=cm_scratch)
# print("RNN From Scratch Confusion Matrix (Requires implementation):")
# print(cm_scratch)

# Kütüphane RNN
cm_lib = confusion_matrix(y_test, y_pred_lib)
disp_lib = ConfusionMatrixDisplay(confusion_matrix=cm_lib)
print("Library RNN Confusion Matrix:")
print(cm_lib)

# Karmaşıklık Matrislerini Çizdirme
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# disp_scratch.plot(ax=axes[0], cmap=plt.cm.Blues)
axes[0].set_title('RNN From Scratch CM\n(Requires Implementation)')
disp_lib.plot(ax=axes[1], cmap=plt.cm.Blues)
axes[1].set_title('Library RNN CM')
plt.tight_layout()
plt.savefig("confusion_matrices.png") # Grafiği kaydet
print("Saved confusion matrices graph to 'confusion_matrices.png'")
# plt.show() # İsteğe bağlı olarak grafikleri göster

print("\n--- Finished ---")