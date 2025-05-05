# main.py

from data import train_data, test_data
from preprocess import build_vocab, preprocess_data
from rnn_from_scratch import RNNFromScratch
from rnn_library import create_library_rnn_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def main():
    # 1. Vocab oluştur
    print("🧠 Vocab hazırlanıyor...")
    vocab = build_vocab({**train_data, **test_data})
    vocab_size = len(vocab)

    # 2. Verileri indeks dizilerine çevir
    X_train_seq, y_train = preprocess_data(train_data, vocab)
    X_test_seq, y_test = preprocess_data(test_data, vocab)

    # 3. Padding
    max_len = max(max(len(seq) for seq in X_train_seq), max(len(seq) for seq in X_test_seq))
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

    # 4. TensorFlow Modelini Eğit
    print("🚀 TensorFlow RNN eğitiliyor...")
    model = create_library_rnn_model(vocab_size=vocab_size, max_length=max_len)
    model.summary()
    model.fit(X_train_pad, y_train, epochs=10, validation_split=0.2, batch_size=8)

    # 5. Tahmin ve değerlendirme
    print("📊 Değerlendirme yapılıyor...")
    loss, acc = model.evaluate(X_test_pad, y_test)
    print(f"Library RNN Test Accuracy: {acc:.4f}")

    # 6. (Opsiyonel) From Scratch RNN
    print("🧪 RNN from scratch eğitimi başlatılmadı (geri yayılım eksik).")

if __name__ == "__main__":
    main()
