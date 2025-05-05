# rnn_library.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def create_library_rnn_model(vocab_size, embedding_dim=16, rnn_units=32, max_length=None):
    """TensorFlow/Keras kullanarak basit bir RNN modeli oluşturur."""
    model = Sequential([
        # Input katmanı (max_length padding sonrası sabit dizi uzunluğu için)
        Input(shape=(max_length,), dtype='int32'),
        # Embedding katmanı: Kelime indekslerini yoğun vektörlere dönüştürür
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True),
        # SimpleRNN katmanı
        SimpleRNN(units=rnn_units),
        # Çıkış katmanı: İkili sınıflandırma için sigmoid aktivasyonu
        Dense(units=1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Örnek Kullanım (train_eval.py içinde kullanılabilir)
# from rnn_library import create_library_rnn_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from data import train_data, test_data
# from preprocess import build_vocab, preprocess_data

# # 1. Veriyi Hazırla
# vocab = build_vocab({**train_data, **test_data})
# X_train_seq, y_train = preprocess_data(train_data, vocab)
# X_test_seq, y_test = preprocess_data(test_data, vocab)

# # 2. Padding (RNN sabit uzunlukta girdi bekler)
# max_len_train = max(len(x) for x in X_train_seq)
# max_len_test = max(len(x) for x in X_test_seq)
# max_length = max(max_len_train, max_len_test) # Genel max uzunluk

# X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
# X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

# # 3. Modeli Oluştur
# model = create_library_rnn_model(vocab_size=len(vocab), max_length=max_length)
# model.summary()

# # 4. Modeli Eğit
# print("Training Library RNN Model...")
# history = model.fit(X_train_pad, y_train, epochs=10, validation_split=0.2, batch_size=8)

# # 5. Modeli Değerlendir
# loss, accuracy = model.evaluate(X_test_pad, y_test)
# print(f"Test Loss: {loss:.4f}")
# print(f"Test Accuracy: {accuracy:.4f}")

# # 6. Tahmin Yap
# sample_text = "this is very good"
# sample_seq = text_to_sequence(sample_text, vocab)
# sample_pad = pad_sequences([sample_seq], maxlen=max_length, padding='post')
# prediction = model.predict(sample_pad)
# print(f"Prediction for '{sample_text}': {prediction[0][0]:.4f} (Positive if > 0.5)")