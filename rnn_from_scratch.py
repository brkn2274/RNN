# rnn_from_scratch.py
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

class RNNFromScratch:
    def __init__(self, vocab_size, hidden_size=64, output_size=1):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Ağırlık matrisleri
        self.Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01

        # Biaslar
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def one_hot(self, idx):
        vec = np.zeros((self.vocab_size, 1))
        vec[idx] = 1
        return vec

    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        self.last_inputs = inputs
        self.last_hs = {0: h}

        # Zaman adımlarında gezin
        for t, idx in enumerate(inputs):
            x = self.one_hot(idx)
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            self.last_hs[t + 1] = h

        y = sigmoid(self.Why @ h + self.by)
        return y

    def backward(self, d_y, learning_rate=0.01):
        n = len(self.last_inputs)
        d_Why = d_y @ self.last_hs[n].T
        d_by = d_y

        d_h = self.Why.T @ d_y

        d_Wxh = np.zeros_like(self.Wxh)
        d_Whh = np.zeros_like(self.Whh)
        d_bh = np.zeros_like(self.bh)

        for t in reversed(range(n)):
            temp = (1 - self.last_hs[t + 1] ** 2) * d_h
            d_bh += temp
            d_Wxh += temp @ self.one_hot(self.last_inputs[t]).T
            d_Whh += temp @ self.last_hs[t].T
            d_h = self.Whh.T @ temp

        # Güncelleme
        self.Wxh -= learning_rate * d_Wxh
        self.Whh -= learning_rate * d_Whh
        self.Why -= learning_rate * d_Why
        self.bh -= learning_rate * d_bh
        self.by -= learning_rate * d_by

    def train(self, sequences, labels, learning_rate=0.01, epochs=10):
        print("Training RNN From Scratch...")
        for epoch in range(epochs):
            total_loss = 0
            for seq, label in zip(sequences, labels):
                y_pred = self.forward(seq)
                loss = -(label * np.log(y_pred) + (1 - label) * np.log(1 - y_pred))
                total_loss += loss
                d_y = y_pred - label
                self.backward(d_y, learning_rate)

            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss[0][0] / len(sequences):.4f}")

    def predict(self, sequence):
        output = self.forward(sequence)
        return int(output > 0.5)
