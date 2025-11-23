# Cyberbullying Detection System

## Overview
This project implements a hybrid cyberbullying detection system combining machine learning and rule-based techniques. It classifies text inputs as cyberbullying or not, with special emphasis on detecting severe threats through pattern matching.

## Features
- Hybrid model architecture:
  - **Machine Learning (SVM with TF-IDF + severity features)**
  - **Rule-based detector for severe abusive language and threats**
- Imbalanced dataset handling using **SMOTE oversampling**
- Data cleaning via heuristic removal of mislabeled samples
- Interactive Streamlit app for single and batch text analysis
- Explainable predictions highlighting influential words and detected severe patterns
- Visualization tools including confusion matrix and probability distributions

## Dataset
- Contains ~47,000 labeled tweets with multiple cyberbullying subcategories
- Data imbalance handled using SMOTE to synthetically balance minority classes without discarding samples
- Cleaning removed ~600 potentially mislabeled non-cyberbullying samples containing severe language patterns

## Model Performance

| Metric                     | Score   |
|----------------------------|---------|
| Overall Accuracy           | 83.2%   |
| Precision (Cyberbullying)   | 93.5%   |
| Recall (Cyberbullying)      | 85.8%   |
| F1-Score (Cyberbullying)    | 89.5%   |
| Precision (Not Cyberbullying)| 49.5%  |
| Recall (Not Cyberbullying)  | 70.1%   |
| F1-Score (Not Cyberbullying)| 58.0%  |
| ROC-AUC Score              | 0.896   |

## Usage

### Training
Run the training script to prepare the hybrid model:
python train_hybrid_model.py

text

### Running the App
Launch the interactive Streamlit interface:
streamlit run app.py

text

## Project Structure
- `train_hybrid_model.py`: Training script including data cleaning, SMOTE oversampling, and model training
- `app.py`: Streamlit app for interactive prediction, batch analysis, and explanations
- `model/`: Directory containing saved model and evaluation artifacts

## Future Enhancements
- Add cross-validation to validate generalization
- Incorporate SHAP explanations for richer interpretability
- Expand rule-based patterns dynamically
- Further improve precision for non-cyberbullying to reduce false positives
- Deploy as scalable API service

## Notes
- Rule-based detection ensures near-zero false negatives on extreme threats (e.g., "kill yourself")
- Model is explainable and practical for real-time content moderation
- Imbalanced data handling essential for balanced learning and performance

---

**This project balances accuracy and interpretability in detecting harmful online language and provides actionable insights via an easy-to-use app interface.**
