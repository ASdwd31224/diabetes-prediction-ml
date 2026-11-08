import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

df = pd.read_csv("diabetes.csv")


# ========================= Data clean ===========================

print(df.head())
print("shape :", df.shape)
print(df.columns.tolist())

print(df.isna().sum())
print("Duplicate rows :")
print(df.duplicated().sum())

print(df.describe().round(2))

# ======================= X and y ==============================

X = df.drop("Outcome", axis=1).copy()
y = df["Outcome"].copy()

# ======================== Zero ============================

zero_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

X[zero_columns] = X[zero_columns].replace(0, float("nan"))

print(X.isnull().sum())

# ========================== Train & Test split ========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(
    "Training rows :",
    len(X_train)
)
print("Testing rows :", len(X_test))


# ========================== Models ========================

models = {
    "Logistic Rgrssion" :
        make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=1000)
        ),
        
        "Decision Tree" :
            make_pipeline(
                SimpleImputer(strategy="median"),
                DecisionTreeClassifier(
                    max_depth=5,
                    random_state=42
                )
            ),
            
        "Random Forest" :
            make_pipeline(
                SimpleImputer(strategy="median"),
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42
                )
            )
}

print("Number of models :", len(models))

# ======================== Train models ========================

for name, model in models.items():
    model.fit(X_train, y_train)
    print(name, "trained successfully.")


# ======================== Compare models ========================

test_results = []

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    test_results.append({
        "Model": name,
        "Accuracy": accuracy_score(
            y_test, y_pred
        ),
        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "F1-score": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(
            y_test, y_prob
        )
    })

test_df = pd.DataFrame(test_results)

print(
    test_df.sort_values(
        by="F1-score",
        ascending=False
    ).round(3).to_string(index=False)
)


# ================================================
# Overfit check
# ================================================

overfit_results = []

for name, model in models.items() :
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_accuracy = accuracy_score(
        y_train,
        train_pred
    )
    
    test_accuracy = accuracy_score(
        y_test,
        test_pred
    )
    
    gap = train_accuracy - test_accuracy
    
    overfit_results.append({
        "Model" : name,
        "Train Accuracy" :
            train_accuracy,
        "Test Accuracy" : 
            test_accuracy,
        "Train-Test Gap" : gap
    })

overfit_df = pd.DataFrame(overfit_results)

print("\nOverfitting check :")
print(overfit_df.round(3).to_string(index=False))


# ========================================
# Cross-Validation
# ========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy" : "accuracy",
    "precision" : "precision",
    "recall" : "recall",
    "f1" : "f1",
    "roc_auc" : "roc_auc"
}

cv_results = []

for name, model in models.items() :
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring
    )
    
    cv_results.append({
        "Model" : name,
        "CV Accuracy" : scores["test_accuracy"].mean(),
        "CV Precision" : scores["test_precision"].mean(),
        "CV Recall" : scores["test_recall"].mean(),
        "CV F1" : scores["test_f1"].mean(),
        "CV ROC-AUC" : scores["test_roc_auc"].mean()
    })
    
cv_df = pd.DataFrame(cv_results)

print("\nCross-Validation Results :")
print(cv_df.round(3).to_string(index=False))


# ===========================================
# Select Best Model
# ===========================================

best_model_name = cv_df.loc[
    cv_df["CV F1"].idxmax(),
    "Model"
]

best_model = models[best_model_name]

print("Selected model :", best_model_name)

print("Mean CV F1 :", round(
    cv_df.loc[
        cv_df["Model"] == best_model_name,
        "CV F1"
    ].iloc[0],
    3
))

# =============================================
# evaluate the selected model
# =============================================

print("\nBest model evaluattio")

best_predict = best_model.predict(X_test)

best_probab = best_model.predict_proba(X_test)[:, 1]

print(
    "Test Accuracy :",
    round(
        accuracy_score(y_test, best_predict),
        3
    )
)

print(
    "Test Precision :",
    round(
        precision_score(y_test, best_predict, zero_division=0),
        3
    )
)

print(
    "Test Recall :",
    round(
        recall_score(y_test, best_predict, zero_division=0),
        3
    )
)

print(
    "Test F1-score :",
    round(
        f1_score(y_test, best_predict, zero_division=0),
        3
    )
)

print(
    "Test ROC-AUC :",
    round(
        roc_auc_score(y_test, best_probab),
        3
    )
)


# ===============================================
# Confusion Matrix
# ===============================================


print("\nConfusion Matrix")

cm = confusion_matrix(
    y_test,
    best_predict,
    labels=[0, 1]
)

print("Rows = actual classes")
print("Columns = predicted classes")
print(cm)

tn, fp, fn, tp = cm.ravel()

print("True Negatives :", tn)
print("False Positives :", fp)
print("False Negatives :", fn)
print("True Positives :", tp)

# ===========================================
# Classification Report
# ===========================================

print("\nClassification Report")

print(
    classification_report(
        y_test,
        best_predict,
        zero_division=0
    )
)

# ===========================================
# Save the best model
# ===========================================

print("\nSave model")

joblib.dump(
    best_model,
    "best_diabetes_model.joblib"
)
print("Model saved successfully!")

# =============================================
# Load the saved model
# =============================================

print("\n Load model")

loaded_model = joblib.load(
    "best_diabetes_model.joblib"
)

print("Model loaded successfully!")


# ============================================================
# 17. PREDICT A NEW SAMPLE
# ============================================================

print("\nNEW SAMPLE PREDICTION")

new_patient = pd.DataFrame(
    [[
        2,
        120,
        70,
        25,
        100,
        28.5,
        0.5,
        30
    ]],
    columns=X.columns
)

prediction = loaded_model.predict(new_patient)[0]

probability = loaded_model.predict_proba(
    new_patient
)[0, 1]

print("Predicted class:", prediction)

print(
    "Estimated model probability:",
    round(probability, 3)
)

if prediction == 1:
    print("Model output: Positive class")
else:
    print("Model output: Negative class")
    

# ============================================================
# 18. PROJECT FINISHED
# ============================================================

print("\n========== PROJECT FINISHED ==========")
