# Arnold Schwarzenegger Facial Recognition

This project applies advanced machine learning techniques to facial recognition, focusing on distinguishing Arnold Schwarzenegger from other individuals using principal component features derived from the Labeled Faces in the Wild (LFW) dataset.

![Facial Recognition](facialrecognition.jpg)

## Project Overview

We play the role of data scientists at a firm specializing in AI-driven facial recognition for high-profile individuals. The goal is to accurately identify Arnold Schwarzenegger in images, enhancing personal security and recognition capabilities.

## Dataset

- **File:** `data/lfw_arnie_nonarnie.csv`
- **Description:** Contains 40 images of Arnold Schwarzenegger and 150 images of other people, represented by principal component features (PC1, PC2, ..., PCN) and a binary label (`1` for Arnold, `0` for others).

| Column Name | Description                                   |
|-------------|-----------------------------------------------|
| PC1, PC2... | Principal components from PCA (image features)|
| Label       | 1 = Arnold Schwarzenegger, 0 = Other          |

## Workflow

1. **Data Preparation:**  
    - Load and split the data into training and test sets, ensuring class balance.

2. **Model Selection:**  
    - Compare Logistic Regression, Random Forest, and SVM classifiers.

3. **Cross-Validation:**  
    - Use 5-fold cross-validation to estimate model performance and reduce overfitting.

4. **Hyperparameter Tuning:**  
    - Perform grid search to find the best parameters for each model.

5. **Evaluation:**  
    - Assess the best model on the test set using accuracy, precision, recall, F1 score, and confusion matrix.

6. **Enhanced Tuning:**  
    - Expand hyperparameter grids and use class weighting to address class imbalance.

## Results

- **Initial Best Model:** Logistic Regression  
  - **Precision:** 1.00  
  - **Recall:** 0.125  
  - **F1 Score:** 0.22  
  - *Very high precision, but low recall (misses most Arnolds).*

- **Enhanced Model:** SVM with class weighting  
  - **Precision:** 0.17  
  - **Recall:** 0.125  
  - **F1 Score:** 0.14  
  - *Takes more risks, but still fails to improve recall.*

## Key Insights

- The models are highly conservative, rarely predicting "Arnold" unless very certain.
- Class imbalance is a major challenge, leading to low recall.
- Enhanced hyperparameter tuning and class weighting did not significantly improve performance.

## Recommendations

- **Address Class Imbalance:** Use oversampling, SMOTE, or more positive examples.
- **Model Complexity:** Try advanced models (e.g., boosting, neural networks).
- **Feature Engineering:** Explore additional features or dimensionality reduction.
- **Threshold Tuning:** Adjust decision thresholds to balance precision and recall.
- **Data Augmentation:** Collect or generate more Arnold images.

## How to Run

1. Clone the repository.
2. Place the dataset in `data/lfw_arnie_nonarnie.csv`.
3. Run the Jupyter notebook step by step to reproduce results and visualizations.
