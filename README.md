# Capstone Project Report 1

## Introduction

### Problem Definition

In Natural Language Processing (NLP), Named Entity Recognition (NER) is a crucial problem.  It is frequently utilized for many different activities, including machine translation, summarization, search, and answering questions.  This project's primary goal is to improve the efficacy and accuracy of NER by implementing machine learning models.  Applications for NER include automated document processing, resume parsing, information retrieval, classification, and customer support.

### Context and Background

A key text processing problem for chatbots, search engines, and automatic document classification is NER.  A recent development is the substitution of machine-learning-based models for conventional rule-based systems, like:
- **CRF (Conditional Random Fields)**: A graphical model with probabilistic labels for sequential data.
- **BiLSTM-CRF**: Bidirectional LSTMs and CRFs are used in a deep learning method for entity recognition.
- **spaCy-based NER**: Neural networks and token-based feature extraction are used in this pipeline architecture.

These techniques aid in increasing generality and accuracy across a range of text corpora.  Studies have demonstrated that NER has been much enhanced by hybrid models that combine statistics and machine learning techniques.

## Objectives and Goals

The following are the project's main goals:
- creating a NER model with several methods (CRF, BiLSTM-CRF, and spaCy models).
- Assessing these models' performance with accuracy metrics like recall, precision, and F1-score.
- Labeled datasets are used for model training and refinement.
- Implementing the top-performing model for practical text extraction applications.

## Summary of Approach

To achieve these objectives, the following approaches will be explored:
- Training a **CRF-based model** on NER using specially-structured features.
- Running a **BiLSTM-CRF** deep learning model.
- Using **spaCy NER pipelines** and customizing them for our specific datasets.
- Evaluating the performance of these models on labeled test data and reporting results.

## Methods

### Data Acquisition and Sources

For this project, a `train_data.pkl` dataset with pre-existing labels was employed.  Each sample represents a text in this text-based dataset, and the relevant entities are tagged.  At the conclusion of this project, we will use the non-labeled dataset to deploy the model and the pre-labeled dataset for classification.

The data-cleaning stage will include:
- Loading and structuring text samples.
- Tokenization and feature extraction.
- Data augmentation and splitting into training and testing sets.

### Mathematical or Statistical Models

A variety of Named Entity Recognition (NER) models are used in this project:
1. **Conditional Random Fields (CRF)**: Sequence labeling is a traditional model that categorizes sequential data points.
2. **Bidirectional LSTM with CRF**: An RNN that processes data sequences over time using past state information.
3. **spaCy-based NER Model**: A specially trained neural network that extracts named and unnamed entities from the text.

### Experimental Design and Analytical Procedures

- **Train-test split**: 90% training and 10% testing.
- **Training steps**:
  - Data preprocessing.
  - Model training with appropriate loss functions.
  - Hyperparameter tuning (e.g., learning rate, dropout rates).
  
- **Evaluation metrics**:
  - Precision, Recall, and F1-score for measuring model performance.
  - Loss reduction curves for monitoring training progress.

### Software and Tools

- **Programming Language**: Python
- **Libraries**: spaCy, Scikit-learn, sklearn-crfsuite, TensorFlow, NumPy, pandas, Matplotlib
- **Computational Resources**: A local machine with GPUs to train new models and retrain existing models.

### Ethical Considerations

Since the project doesn’t use confidential personal data, there are minimal ethical concerns. However, when applying this technology in real-world applications, it's essential to consider how certain biases in the training data could impact the model's performance.

## Results

### Model Evaluation Report

#### Overall Performance

The evaluation metrics for different models are as follows:

**General Model Evaluation Results**:

| Entity Type       | Precision | Recall | F1-score | Support |
|-------------------|-----------|--------|----------|---------|
| College Name      | 0.81      | 0.54   | 0.65     | 106     |
| Companies worked at| 0.85     | 0.23   | 0.36     | 101     |
| Degree            | 0.83      | 0.81   | 0.82     | 91      |
| Designation       | 0.90      | 0.28   | 0.43     | 128     |
| Email Address     | 0.68      | 0.46   | 0.55     | 28      |
| Graduation Year   | 0.00      | 0.00   | 0.00     | 14      |
| Location          | 0.47      | 0.42   | 0.44     | 38      |
| Name              | 0.97      | 0.84   | 0.90     | 43      |
| O                 | 0.93      | 0.97   | 0.95     | 11921   |
| Skills            | 0.37      | 0.23   | 0.29     | 811     |
| Years of Experience| 0.00     | 0.00   | 0.00     | 12      |

#### CRF Model Performance on Test Set:

| Entity Type       | Precision | Recall | F1-score | Support |
|-------------------|-----------|--------|----------|---------|
| College Name      | 0.79      | 0.43   | 0.56     | 126     |
| Companies worked at| 0.75     | 0.35   | 0.48     | 138     |
| Degree            | 0.83      | 0.67   | 0.74     | 116     |
| Designation       | 0.56      | 0.57   | 0.57     | 150     |
| Email Address     | 0.79      | 0.52   | 0.63     | 21      |
| Graduation Year   | 0.17      | 0.07   | 0.10     | 14      |
| Location          | 0.62      | 0.37   | 0.47     | 54      |
| Name              | 0.90      | 0.86   | 0.88     | 42      |
| O                 | 0.95      | 0.99   | 0.97     | 15969   |
| Skills            | 0.92      | 0.39   | 0.55     | 990     |

**Weighted F1 Score**: 0.9295

#### Training Results

**Training performance across epochs for the BiLSTM model**:
- Accuracy improved from **61.51% (Epoch 1)** to **96.98% (Epoch 5)**.
- Loss decreased from **2.4127 (Epoch 1)** to **0.1787 (Epoch 5)**.
- The model achieved validation accuracy of **96.00%**.

#### Sample Predictions (BiLSTM Model)

**Predicted Entities**:
- UNKNOWN - John
- College Name - Doe
- College Name - is
- College Name - a
- College Name - Software
- College Name - Engineer
- College Name - at
- College Name - Google.

### Model Comparison

---![image](https://github.com/user-attachments/assets/7551cfd2-f25c-4c21-8e73-ae338e6ba245)


## Limitations of Results

Potential limitations include:
- **Data bias** due to small sample size.
- **Overfitting** issues in deep learning models.
- **Challenges in handling ambiguous text contexts**.




