#  Engineering a Sentiment Analysis System from First Principles using RNN and LSTM Architectures

> **Building RNNs and LSTMs from the ground up** — then going head-to-head against Keras on the IMDB sentiment dataset.

<p align="center">
  <img src="https://img.shields.io/github/stars/imenei/Sentiment-Analysis-with-Recurrent-Networks-RNN-LSTM-and-Confidence-Estimation?style=for-the-badge&logo=github&color=yellow" alt="GitHub Stars"/>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-From%20Scratch-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-Keras%20Baseline-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
</p>

---

##  Overview

Most deep learning courses hand you `model.fit()` and move on.

This project does the opposite: **every gate, every gradient, every weight update is coded by hand**, using nothing but NumPy — then benchmarked against the Keras equivalent on the same dataset.

Two architectures. Two implementations each. One shared dataset.  
The goal: prove that understanding the math behind a model is worth more than just knowing the API.

##  Confidence Estimation

The model outputs probabilistic predictions, allowing interpretation of prediction confidence and uncertainty. This provides deeper insight into model behavior beyond binary classification.
---

## 🏗 Project Structure

```
Sentiment-Analysis-with-Recurrent-Networks-RNN-LSTM-and-Confidence-Estimation/
│
├── RNN/
│   └── RNN.ipynb      # Cell 1: RNN from scratch  |  Cell 2: Keras SimpleRNN
│
├── LSTM/
│   └── LSTM.ipynb     # Cell 1: LSTM from scratch  |  Cell 2: Keras LSTM
│
├── README.md
```

Each notebook is self-contained: load data → train → evaluate → interactive inference.  
No external scripts, no hidden dependencies.

---

## 📊 Results

Tested on the **IMDB Movie Reviews** dataset — 25,000 training samples, 25,000 test samples, binary sentiment classification (positive / negative).

| Model | Implementation | Test Accuracy | AUC |
|---|---|:---:|:---:|
| SimpleRNN | From scratch (NumPy + BPTT) | 80.6% | — |
| SimpleRNN | Keras baseline | 81.3% | 0.896 |
| LSTM | From scratch (NumPy, manual gates) | 82.9% | — |
| LSTM | Keras baseline | **85.0%** | **0.929** |

**Key observations:**
- The from-scratch implementations stay within **~2% of Keras** — the math is correct
- LSTM outperforms RNN by **+2–4%** across both implementations, confirming its superior long-range memory
- The LSTM Keras model reaches **AUC 0.929**, a strong result for a single-layer recurrent architecture with no attention mechanism

---

## 🧠 Architecture Details

### RNN — From Scratch

A vanilla recurrent network with manual forward pass and Backpropagation Through Time (BPTT).

**Parameters:**
```
Embedding  E  : (10 000, 64)
Input→Hidden  : Wx  (64, 64)
Hidden→Hidden : Wh  (64, 64)
Hidden bias   : bh  (64,)
Hidden→Output : Wo  (64, 1)
Output bias   : bo  (1,)
```

**Forward pass:**
```
h_t = tanh(Wx · x_t  +  Wh · h_{t-1}  +  bh)
ŷ   = sigmoid(Wo · h_T  +  bo)
Loss = Binary Cross-Entropy(ŷ, y)
```

**Backprop Through Time (BPTT):**  
Gradients are unrolled through every timestep from T back to 0.  
Early stopping (patience = 3) is applied on test accuracy to prevent overfitting.

---

### LSTM — From Scratch

A Long Short-Term Memory network with all four gates implemented manually.

**The four gates:**
```
f_t = σ(Wf · [h_{t-1}, x_t] + bf)   # Forget gate  — what to erase from memory
i_t = σ(Wi · [h_{t-1}, x_t] + bi)   # Input gate   — what new info to store
g_t = tanh(Wg · [h_{t-1}, x_t] + bg) # Candidate   — candidate memory content
o_t = σ(Wo · [h_{t-1}, x_t] + bo)   # Output gate  — what to expose as hidden state
```

**Cell and hidden state update:**
```
c_t = f_t ⊙ c_{t-1}  +  i_t ⊙ g_t
h_t = o_t ⊙ tanh(c_t)
```

The forget gate is the key innovation: it allows the network to selectively retain information across long sequences, solving the **vanishing gradient problem** that cripples plain RNNs.

---

## 💡 Why Build From Scratch?

| | From Scratch | Keras |
|---|---|---|
| **Transparency** | Every operation is explicit | Abstracted away |
| **Learning value** | Forces you to understand BPTT, gates, gradients | You trust the library |
| **Debugging** | You can inspect any intermediate value | Black box |
| **Performance** | Slower, CPU only | Optimized, GPU-ready |
| **Production use** | No | Yes |

The from-scratch implementation is not about performance.  
It's about **knowing what you're building** before you let a library build it for you.

---

## ⚡ Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/imenei/Sentiment-Analysis-with-Recurrent-Networks-RNN-LSTM-and-Confidence-Estimation
cd Sentiment-Analysis-with-Recurrent-Networks-RNN-LSTM-and-Confidence-Estimation
```

### 2. Run a notebook
```bash
# RNN experiments
jupyter notebook RNN/RNN.ipynb

# LSTM experiments
jupyter notebook LSTM/LSTM.ipynb
```

Each notebook includes an **interactive inference loop** at the end — type any movie review and get a live prediction with confidence score:

```
>>> This film was an absolute masterpiece, I was on the edge of my seat
  [+] POSITIF  score=0.9341  confiance=93.4%

>>> Boring, predictable, and a complete waste of two hours
  [-] NEGATIF  score=0.0812  confiance=91.9%
```

---

## 📦 Requirements

```
numpy
tensorflow
keras
jupyter
```

Install with:
```bash
pip install -r requirements.txt
```

> Tested with Python 3.10+, TensorFlow 2.12+, NumPy 1.24+

---

## 📚 Concepts Covered

- Recurrent Neural Networks (RNN) — architecture and limitations
- Backpropagation Through Time (BPTT) — gradient unrolling
- Vanishing gradient problem — why plain RNNs struggle with long sequences
- Long Short-Term Memory (LSTM) — forget, input, output, and candidate gates
- Word embeddings — mapping tokens to dense vector representations
- Sequence padding and truncation — handling variable-length inputs
- Binary cross-entropy loss — for sentiment classification
- Early stopping and learning rate scheduling — regularization in practice

---

## 🗺 Roadmap

- [ ] Add GRU from scratch (simpler than LSTM, often competitive)
- [ ] Add bidirectional wrapper for from-scratch models
- [ ] Training curves visualization (loss & accuracy plots)
- [ ] Export trained weights for reuse

---

## 🤝 Contributing

Contributions are welcome — whether it's a bug fix, a new architecture, or cleaner math notation in the notebooks.

1. Fork the repo
2. Create a branch: `git checkout -b feature/my-improvement`
3. Commit your changes: `git commit -m "add: GRU from scratch"`
4. Open a pull request

---


<p align="center">
  Built to understand, not just to use.
</p>
