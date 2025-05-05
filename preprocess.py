# preprocess.py
import numpy as np

def build_vocab(data):
    """Veri setindeki tüm benzersiz kelimeleri içeren bir sözlük oluşturur."""
    vocab = set()
    for text in data.keys():
        for word in text.split(' '):
            vocab.add(word)
    return {word: i for i, word in enumerate(vocab)}

def text_to_sequence(text, vocab):
    """Metni kelime indekslerinden oluşan bir diziye dönüştürür."""
    return [vocab[word] for word in text.split(' ') if word in vocab]

def preprocess_data(data, vocab):
    """Veri setini metin ve etiketler olarak ayırır ve metinleri dizilere dönüştürür."""
    sequences = [text_to_sequence(text, vocab) for text in data.keys()]
    labels = np.array(list(data.values())).astype(int) # True/False -> 1/0
    return sequences, labels

# Örnek Kullanım (train_eval.py içinde kullanılabilir)
# from data import train_data, test_data
# from preprocess import build_vocab, preprocess_data

# vocab = build_vocab({**train_data, **test_data}) # Eğitim ve test verisindeki tüm kelimeler
# X_train_seq, y_train = preprocess_data(train_data, vocab)
# X_test_seq, y_test = preprocess_data(test_data, vocab)

# print("Vocabulary size:", len(vocab))
# print("Sample sequence:", X_train_seq[0])
# print("Sample label:", y_train[0])