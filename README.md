# Credit Card Fraud Detection: Sampling Techniques Analysis

This project investigates the impact of different sampling techniques on the performance of various machine learning models. Using a balanced credit card fraud dataset, we apply five distinct sampling methods to create training sets and evaluate five different classifiers.

## Methodology

**Data Balancing**: The original dataset was balanced using Random Over-Sampling to handle the class imbalance (initially 763 legitimate vs 9 fraud cases).

**Sample Size**: The target sample size was calculated using the Z-score formula for a 95% confidence level and 5% margin of error.

## Sampling Techniques (Columns)

The following sampling techniques were applied to generate the training data:

- **Sampling1 (Simple Random Sampling)**: Random selection of samples from the population.
- **Sampling2 (Systematic Sampling)**: Selection of samples at a regular interval (every $k^{th}$ item).
- **Sampling3 (Stratified Sampling)**: Sampling that maintains the proportion of classes (Fraud vs. Non-Fraud) found in the original population.
- **Sampling4 (Cluster Sampling)**: Population divided into clusters; specific clusters are selected randomly.
- **Sampling5 (Bootstrap Sampling)**: Random sampling with replacement.

## Machine Learning Models (Rows)

Five different classifiers were trained on the samples:

- M1: Logistic Regression
- M2: Decision Tree
- M3: Random Forest
- M4: Support Vector Machine (SVM)
- M5: K-Nearest Neighbors (KNN)

Results
The table below shows the accuracy achieved by each model (M1-M5) using each sampling technique.

| Model              | Simple Random | Systematic | Stratified | Cluster | Bootstrap |
| ------------------ | ------------- | ---------- | ---------- | ------- | --------- |
| M1 (Logistic)      | 0.9221        | 0.9481     | 0.8961     | 0.9079  | 0.9610    |
| M2 (Decision Tree) | 0.9870        | 1.0000     | 0.9870     | 0.9737  | 0.9610    |
| M3 (Random Forest) | 1.0000        | 1.0000     | 0.9870     | 1.0000  | 1.0000    |
| M4 (SVM)           | 0.9091        | 0.9481     | 0.9091     | 0.9079  | 0.8961    |
| M5 (KNN)           | 0.9740        | 0.9870     | 0.8961     | 0.9737  | 0.9481    |

**Best Performing Model**: M3 (Random Forest) consistently achieved the highest accuracy, reaching 100% (1.0) across almost all sampling techniques. This is expected given Random Forest's robustness to variance and the nature of over-sampled datasets.

**Best Sampling Technique**: Systematic Sampling (Sampling2) appeared to yield the highest average accuracy across all models, boosting even the weaker classifiers like Logistic Regression and SVM.

**Consistency**: Bootstrap Sampling (Sampling5) provided very consistent high-performance results for Logistic Regression (M1), outperforming other techniques for that specific model.

**SVC Performance**: Support Vector Machine (M4) generally had the lowest accuracy among the group, particularly with Cluster and Simple Random sampling, suggesting it may require more careful hyperparameter tuning or feature scaling for this specific dataset.
