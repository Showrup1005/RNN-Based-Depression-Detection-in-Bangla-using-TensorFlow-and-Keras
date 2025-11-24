# RNN-Based-Depression-Detection-in-Bangla-using-TensorFlow-and-Keras
A deep learning-based system for classifying depression severity levels from textual data using multiple recurrent neural network architectures.
## 📋 Project Overview
This project implements and compares three different neural network models (SimpleRNN, LSTM, and Bidirectional LSTM) for classifying depression severity into five categories: non-depressive, mild, moderate, depressive, and severe. The system processes Bengali text data through a comprehensive preprocessing pipeline and employs advanced deep learning techniques for mental health assessment.
## 🏗️ Model Architectures
### 1. SimpleRNN
   - Embedding Layer (128 dimensions)

   - SimpleRNN Layer (64 units) with dropout (0.3)

   - Dense Layer (64 units) with ReLU activation

   - Dropout Layer (0.3)

   - Output Layer (5 units) with softmax activation

### 2. LSTM
   - Embedding Layer (128 dimensions)

   - LSTM Layer (64 units) with dropout (0.3)

   - Dense Layer (64 units) with ReLU activation

   - Dropout Layer (0.3)

   - Output Layer (5 units) with softmax activation

### 3. Bidirectional LSTM
   - Embedding Layer (128 dimensions)

   - Bidirectional LSTM Layer (64 units) with dropout (0.3)

   - Dense Layer (64 units) with ReLU activation

   - Dropout Layer (0.3)

   - Output Layer (5 units) with softmax activation
## 📊 Dataset
The dataset contains 9,449 text samples distributed across five depression severity classes:
| Class          | Total Instances | Training | Validation | Test |
|----------------|-----------------|----------|------------|------|
| non_depressive | 5,033           | 4,026    | 504        | 503  |
| mild           | 2,083           | 1,666    | 208        | 209  |
| moderate       | 1,340           | 1,072    | 134        | 134  |
| severe         | 670             | 536      | 67         | 67   |
| very_severe    | 323             | 259      | 32         | 32   |
| **Total**      | **9,449**       | **7,559**| **945**    | **945** |
## ⚙️ Installation
### Prerequisites
   - Python 3.8+

   - TensorFlow 2.6+
     
   - scikit-learn
     
   - pandas

   - numpy

   - NLTK
### Setup
```bash
git clone https://github.com/Showrup1005/RNN-Based-Depression-Detection-in-Bangla-using-TensorFlow-and-Keras
cd RNN-Based-Depression-Detection-in-Bangla-using-TensorFlow-and-Keras
pip install -r requirements.txt
```
## 📈 Performance Results
### Overall Metrics
| Model   | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| RNN     | 59.79%   | 45.59%    | 59.78% | 51.70%   |
| LSTM    | 62.96%   | 54.38%    | 62.96% | 55.95%   |
| Bi-LSTM | 60.21%   | 58.27%    | 60.21% | 58.95%   |
### Key Findings
 
 - LSTM achieved the highest overall accuracy (62.96%)

- Bi-LSTM showed the most balanced performance across precision and recall

- All models struggled with class imbalance, particularly for minority classes

- Bi-LSTM demonstrated superior capability in detecting severe depression cases
## 🔧 Technical Specifications
## Hardware
- Processor: Intel Core i5-1135G7

- GPU: NVIDIA GeForce MX350 (2GB GDDR5)

- RAM: 8GB DDR4

## Software
- Python 3.8.10

- TensorFlow 2.6.0

- Keras 2.6.0

- scikit-learn 0.24.2

```
depression-severity-classification/
│
├── README.md
├── requirements.txt
├── Bangla_stopwords.txt
│
├── notebook/
│   └── CODE_1005_1010_1012.ipynb
│
├── models/
│   ├── best_model_rnn.weights.h5
│   ├── best_model_lstm.weights.h5
│   └── best_model_bilstm.weights.h5
│   
└── results/
    ├── confusion_matrices/
    │   ├── rnn_confusion_matrix.png
    │   ├── lstm_confusion_matrix.png
    │   └── bilstm_confusion_matrix.png
    │
    ├── loss_curves/
    │    ├── rnn_loss_curve.png
    │    ├── lstm_loss_curve.png
    │    └── bilstm_loss_curve.png
    │ 
    └── accuracy_curves/
        ├── rnn_accuracy_curves.png
        ├── lstm_accuracy_curves.png
        └── bilstm_accuracy_curves.png
```
