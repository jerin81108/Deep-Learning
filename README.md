# Deep Learning Laboratory (23ADL51) - Complete Experiments

All 12 laboratory experiments from the manual have been minimized, optimized for quick and reproducible execution, and saved in this directory:

| Experiment | Title | File | Description & Architecture |
| :--- | :--- | :--- | :--- |
| **Ex. 1** | Solving XOR Problem using DNN | [`exp1.py`](./exp1.py) | 2-2-1 Deep Neural Network using pure NumPy with backpropagation and sigmoid activation. |
| **Ex. 2** | Character Recognition using CNN | [`exp2.py`](./exp2.py) | 2D CNN with Conv2D, MaxPool2D, Dropout, Dense for 26-class handwritten alphabet classification. |
| **Ex. 3** | Face Recognition using CNN | [`exp3.py`](./exp3.py) | LFW face dataset classifier with 3 Conv2D layers, MaxPool2D, and identity prediction. |
| **Ex. 4** | Language Modeling using RNN | [`exp4.py`](./exp4.py) | PyTorch Character-level Recurrent Neural Network for nationality-conditioned name generation. |
| **Ex. 5** | Sentiment Analysis using LSTM | [`exp5.py`](./exp5.py) | Keras IMDB movie review sentiment classifier using Word Embeddings and LSTM. |
| **Ex. 6** | POS Tagging using Seq2Seq | [`exp6.py`](./exp6.py) | Sequence-to-Sequence Part-of-Speech tagging with Embedding, LSTM, and TimeDistributed Dense. |
| **Ex. 7** | Machine Translation using Encoder-Decoder | [`exp7.py`](./exp7.py) | English-to-French translation using Seq2Seq LSTM Encoder-Decoder with state passing. |
| **Ex. 8** | Image Augmentation using GANs | [`exp8.py`](./exp8.py) | Deep Convolutional Generative Adversarial Network (DCGAN) for MNIST synthetic digits. |
| **Ex. 9** | Video Captioning / Action Recognition | [`exp9.py`](./exp9.py) | Hybrid CNN-LSTM video classification extracting frame features and temporal sequences. |
| **Ex. 10** | Sign Language Recognition & Speech Output | [`exp10.py`](./exp10.py) | MediaPipe hand landmark extraction + LSTM gesture classifier with text-to-speech engine. |
| **Ex. 11** | Hand Gesture Controlled Arcade Game | [`exp11.py`](./exp11.py) | Interactive Pygame paddle game controlled in real-time via index finger tracking. |
| **Ex. 12** | Automated Student Attendance System | [`exp12.py`](./exp12.py) | Full Tkinter desktop GUI for automated lab terminal allocation and attendance verification. |

## Execution Instructions

### Option 1: Run Locally
Install the required packages:
```bash
pip install -r requirements.txt
```
Run any experiment:
```bash
python exp1.py
```

### Option 2: Run in Google Colab (Recommended)
1. Open [Google Colab](https://colab.research.google.com).
2. Create a new notebook and set Runtime to **T4 GPU** (`Runtime > Change runtime type > T4 GPU`).
3. Copy-paste the desired script into a code cell and click **Run**.

