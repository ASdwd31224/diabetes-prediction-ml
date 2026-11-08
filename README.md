Diabetes Prediction Using Machine Learning

<p align="center">
  A Machine Learning Classification Project
</p>---

About The Project

This project focuses on predicting diabetes using machine learning classification algorithms.

It follows a complete machine learning workflow, including data preprocessing, model training, cross-validation, model comparison, and performance evaluation.

The main objective is to compare different classification algorithms and select the best-performing model based on cross-validation results.

Machine Learning Models

Three classification algorithms are implemented:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

Project Workflow

1. Data Loading and Exploration
2. Data Quality Inspection
3. Handling Missing and Invalid Values
4. Feature and Target Separation
5. Train-Test Split
6. Data Preprocessing Using Pipelines
7. Stratified 5-Fold Cross-Validation
8. Model Training
9. Model Performance Comparison
10. Overfitting Analysis
11. Best Model Selection
12. Final Model Evaluation
13. Confusion Matrix Analysis
14. Classification Report
15. Model Saving and Loading
16. Prediction on New Data

Technologies Used

- Python
- Pandas
- Scikit-learn
- Joblib

Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

Project Structure

diabetes-prediction-ml/
│
├── src/
│   └── train.py
│
├── data/
├── models/
│
├── requirements.txt
├── README.md
└── .gitignore

Installation

Clone this repository:

git clone https://github.com/ASdwd31224/diabetes-prediction-ml.git

Navigate to the project directory:

cd diabetes-prediction-ml

Install the required libraries:

pip install -r requirements.txt

Place the dataset in the "data" directory and run:

python src/train.py

Results

The project compares three machine learning models using cross-validation and evaluates the selected model on a separate test dataset.

Actual evaluation results will be documented here after the final experiment.

Future Improvements

- Hyperparameter Optimization
- Feature Importance Analysis
- Additional Machine Learning Models
- Model Interpretability
- Improved Prediction Performance

Disclaimer

This project is intended for educational and research purposes only. It is not a medical diagnostic tool and should not be used to make healthcare decisions.

---

Developed by ASdwd31224
