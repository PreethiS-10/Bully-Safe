Cyberbullying Detection Model - README
Overview
This project implements a hybrid cyberbullying detection system that combines a machine learning model with a rule-based severe bullying detector. The system classifies text for cyberbullying with high accuracy and robust handling of severe threats.

Dataset and Imbalance Handling
The training dataset is significantly imbalanced with many more bullying samples than non-bullying.

Class imbalance was addressed by applying SMOTE (Synthetic Minority Oversampling Technique) to generate synthetic samples of the minority class, improving the model’s ability to learn from limited non-bullying examples without discarding majority class data.

Additionally, domain-specific pattern-based cleaning was applied to remove mislabeled samples from the dataset to improve data quality.

Model Performance Summary
Metric	Score
Accuracy	83.2%
Precision (Cyberbullying)	93.5%
Recall (Cyberbullying)	85.8%
F1-Score (Cyberbullying)	89.5%
ROC-AUC	0.896
Key Strengths
Effectively detects severe bullying via rule-based overrides.

Balances precision and recall on imbalanced dataset.

Provides explainable feature contributions.

Designed for real-world moderation scenarios.

Usage
Train the model with the provided script (includes imbalance handling).

Use the Streamlit app for analysis, visualization, and batch processing.
