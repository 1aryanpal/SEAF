import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics 
import warnings
warnings.filterwarnings('ignore')


# loading data
data = pd.read_csv("data/phishing.csv")
data.head()

data.shape
data.columns
data.info()
data.nunique()
data = data.drop(['Index'],axis = 1)

#description of dataset
data.describe().T
# Splitting the dataset into dependant and independant fetature

X = data.drop(["class"],axis =1)
y = data["class"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
X_train.shape, y_train.shape, X_test.shape, y_test.shape

ML_Model = []
accuracy = []
f1_score = []
recall = []
precision = []

#function to call for storing the results
def storeResults(model, a,b,c,d):
  ML_Model.append(model)
  accuracy.append(round(a, 3))
  f1_score.append(round(b, 3))
  recall.append(round(c, 3))
  precision.append(round(d, 3))

  # Linear regression model 
from sklearn.linear_model import LogisticRegression
#from sklearn.pipeline import Pipeline

# instantiate the model
log = LogisticRegression()

# fit the model 
log.fit(X_train,y_train)

#predicting the target value from the model for the samples

y_train_log = log.predict(X_train)
y_test_log = log.predict(X_test)

acc_train_log = metrics.accuracy_score(y_train,y_train_log)
acc_test_log = metrics.accuracy_score(y_test,y_test_log)
print("Logistic Regression : Accuracy on training Data: {:.3f}".format(acc_train_log))
print("Logistic Regression : Accuracy on test Data: {:.3f}".format(acc_test_log))
print()

f1_score_train_log = metrics.f1_score(y_train,y_train_log)
f1_score_test_log = metrics.f1_score(y_test,y_test_log)
print("Logistic Regression : f1_score on training Data: {:.3f}".format(f1_score_train_log))
print("Logistic Regression : f1_score on test Data: {:.3f}".format(f1_score_test_log))
print()

recall_score_train_log = metrics.recall_score(y_train,y_train_log)
recall_score_test_log = metrics.recall_score(y_test,y_test_log)
print("Logistic Regression : Recall on training Data: {:.3f}".format(recall_score_train_log))
print("Logistic Regression : Recall on test Data: {:.3f}".format(recall_score_test_log))
print()

precision_score_train_log = metrics.precision_score(y_train,y_train_log)
precision_score_test_log = metrics.precision_score(y_test,y_test_log)
print("Logistic Regression : precision on training Data: {:.3f}".format(precision_score_train_log))
print("Logistic Regression : precision on test Data: {:.3f}".format(precision_score_test_log))

#computing the classification report of the model
print(metrics.classification_report(y_test, y_test_log))

# Gradient Boosting Classifier Model
from sklearn.ensemble import GradientBoostingClassifier

# instantiate the model
gbc = GradientBoostingClassifier(max_depth=4,learning_rate=0.7)

# fit the model 
gbc.fit(X_train,y_train)

#predicting the target value from the model for the samples
y_train_gbc = gbc.predict(X_train)
y_test_gbc = gbc.predict(X_test)

#computing the accuracy, f1_score, Recall, precision of the model performance

acc_train_gbc = metrics.accuracy_score(y_train,y_train_gbc)
acc_test_gbc = metrics.accuracy_score(y_test,y_test_gbc)
print("Gradient Boosting Classifier : Accuracy on training Data: {:.3f}".format(acc_train_gbc))
print("Gradient Boosting Classifier : Accuracy on test Data: {:.3f}".format(acc_test_gbc))
print()

f1_score_train_gbc = metrics.f1_score(y_train,y_train_gbc)
f1_score_test_gbc = metrics.f1_score(y_test,y_test_gbc)
print("Gradient Boosting Classifier : f1_score on training Data: {:.3f}".format(f1_score_train_gbc))
print("Gradient Boosting Classifier : f1_score on test Data: {:.3f}".format(f1_score_test_gbc))
print()

recall_score_train_gbc = metrics.recall_score(y_train,y_train_gbc)
recall_score_test_gbc =  metrics.recall_score(y_test,y_test_gbc)
print("Gradient Boosting Classifier : Recall on training Data: {:.3f}".format(recall_score_train_gbc))
print("Gradient Boosting Classifier : Recall on test Data: {:.3f}".format(recall_score_test_gbc))
print()

precision_score_train_gbc = metrics.precision_score(y_train,y_train_gbc)
precision_score_test_gbc = metrics.precision_score(y_test,y_test_gbc)
print("Gradient Boosting Classifier : precision on training Data: {:.3f}".format(precision_score_train_gbc))
print("Gradient Boosting Classifier : precision on test Data: {:.3f}".format(precision_score_test_gbc))

#computing the classification report of the model

print(metrics.classification_report(y_test, y_test_gbc))

training_accuracy = []
test_accuracy = []
# try learning_rate from 0.1 to 0.9
depth = range(1,10)
for n in depth:
    forest_test =  GradientBoostingClassifier(learning_rate = n*0.1)

    forest_test.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(forest_test.score(X_train, y_train))
    # record generalization accuracy
    test_accuracy.append(forest_test.score(X_test, y_test))
    

#plotting the training & testing accuracy for n_estimators from 1 to 50
plt.figure(figsize=None)
plt.plot(depth, training_accuracy, label="training accuracy")
plt.plot(depth, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")  
plt.xlabel("learning_rate")
plt.legend();

training_accuracy = []
test_accuracy = []
# try learning_rate from 0.1 to 0.9
depth = range(1,10,1)
for n in depth:
    forest_test =  GradientBoostingClassifier(max_depth=n,learning_rate = 0.7)

    forest_test.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(forest_test.score(X_train, y_train))
    # record generalization accuracy
    test_accuracy.append(forest_test.score(X_test, y_test))
    

#plotting the training & testing accuracy for n_estimators from 1 to 50
plt.figure(figsize=None)
plt.plot(depth, training_accuracy, label="training accuracy")
plt.plot(depth, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")  
plt.xlabel("max_depth")
plt.legend();

plt.legend();

# Save the trained Gradient Boosting model
import pickle
# --- Save confusion matrices for both models ---
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

# Ensure artifacts directory exists
os.makedirs("artifacts", exist_ok=True)

def save_confusion_matrix(y_true, y_pred, model_name, labels=['Legitimate', 'Phishing']):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    plt.figure(figsize=(5, 4))
    disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(f"artifacts/confusion_matrix_{model_name.lower()}.png")
    plt.close()

# Save confusion matrices
save_confusion_matrix(y_test, y_test_log, "LogisticRegression")
save_confusion_matrix(y_test, y_test_gbc, "GradientBoosting")

print("✅ Confusion matrices saved successfully in 'artifacts/' folder.")

pickle.dump(gbc, open("models/model.pkl", "wb"))
print("✅ Model saved successfully at 'models/model.pkl'")








# ============================
# FULL, CRASH-PROOF TRAINING PIPELINE
# (DecisionTree, RandomForest, SVM, KNN, NaiveBayes, XGBoost optional)
# ============================

import matplotlib
matplotlib.use("Agg")  # Prevent Tkinter crash

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import pickle, os, warnings
warnings.filterwarnings("ignore")

# -----------------------
# Make sure folders exist
# -----------------------
os.makedirs("models", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# -----------------------
# SAFE metric lists
# -----------------------
metric_model = []
metric_accuracy = []
metric_f1 = []
metric_recall = []
metric_precision = []
metric_auc = []
metric_time = []

# -----------------------
# Helper function
# -----------------------
def safe_store_results(model_name, acc, f1, rec, prec, auc, elapsed):
    metric_model.append(model_name)
    metric_accuracy.append(round(acc, 4))
    metric_f1.append(round(f1, 4))
    metric_recall.append(round(rec, 4))
    metric_precision.append(round(prec, 4))
    metric_auc.append(auc)
    metric_time.append(round(elapsed, 4))


def save_classification_report_file(y_true, y_pred, model_name):
    report = classification_report(y_true, y_pred, digits=4)
    with open(f"reports/{model_name}_classification_report.txt", "w") as f:
        f.write(report)


def save_confusion_matrix_file(y_true, y_pred, model_name, labels=['Legitimate','Phishing']):
    cm = confusion_matrix(y_true, y_pred)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    plt.figure(figsize=(5,4))
    disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(f"artifacts/confusion_matrix_{model_name}.png")
    plt.close()


def save_roc_curve(y_true, y_scores, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.4f})")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/roc_{model_name}.png")
    plt.close()
    return auc

# -----------------------
# Model constructors
# -----------------------
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

model_constructors = {
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(),
    "NaiveBayes": GaussianNB()
}

# -----------------------
# Grid search grids
# -----------------------
grids = {
    "DecisionTree": {"max_depth": [3,5,7,None], "criterion": ["gini","entropy"]},
    "RandomForest": {"n_estimators": [100,200], "max_depth": [5,10,None]},
    "SVM": {"C":[0.1,1,10], "kernel":["rbf","linear"]},
    "KNN": {"n_neighbors":[3,5,7], "weights":["uniform","distance"]},
}

RUN_GRIDSEARCH = True

# -----------------------
# Storage
# -----------------------
detailed_results = []
combined_roc_curves = []

# -----------------------
# Training loop
# -----------------------
for model_name, constructor in model_constructors.items():
    print(f"\n========== {model_name} ==========")
    start_time = time.time()

    # Grid search
    if RUN_GRIDSEARCH and model_name in grids:
        print(" -> Running GridSearchCV ...")
        gs = GridSearchCV(constructor, grids[model_name], cv=4, scoring="accuracy", n_jobs=-1)
        gs.fit(X_train, y_train)
        model = gs.best_estimator_
        best_params = gs.best_params_
        print(" Best params:", best_params)
    else:
        model = constructor
        model.fit(X_train, y_train)
        best_params = None

    # Predictions
    y_pred = model.predict(X_test)

    # Scores for ROC
    y_scores = None
    if hasattr(model, "predict_proba"):
        try:
            y_scores = model.predict_proba(X_test)[:,1]
        except:
            pass
    if y_scores is None and hasattr(model, "decision_function"):
        try:
            y_scores = model.decision_function(X_test)
        except:
            pass

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_scores) if y_scores is not None else None

    elapsed = time.time() - start_time

    # Store
    safe_store_results(model_name, acc, f1, rec, prec, auc, elapsed)
    detailed_results.append({
        "Model": model_name,
        "Accuracy": acc,
        "F1": f1,
        "Recall": rec,
        "Precision": prec,
        "AUC": auc,
        "Time_sec": round(elapsed,2),
        "BestParams": best_params
    })

    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | Recall: {rec:.4f} | Precision: {prec:.4f} | AUC: {auc}")
    print(f"Time: {elapsed:.2f}s")

    # Save model
    pickle.dump(model, open(f"models/{model_name}.pkl", "wb"))

    # Save confusion matrix + classification report
    save_confusion_matrix_file(y_test, y_pred, model_name)
    save_classification_report_file(y_test, y_pred, model_name)

    # ROC
    if y_scores is not None:
        auc_val = save_roc_curve(y_test, y_scores, model_name)
        combined_roc_curves.append((model_name, y_scores, auc_val))

# -----------------------
# Combined ROC Plot
# -----------------------
if combined_roc_curves:
    plt.figure(figsize=(8,6))
    for (name, ys, aucv) in combined_roc_curves:
        fpr, tpr, _ = roc_curve(y_test, ys)
        plt.plot(fpr, tpr, label=f"{name} (AUC={aucv:.4f})")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curves - All Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/roc_all_models.png")
    plt.close()

# -----------------------
# Comparison Summary
# -----------------------
df_compare = pd.DataFrame(detailed_results).sort_values(by="Accuracy", ascending=False)
df_compare.to_csv("artifacts/model_comparison_full.csv", index=False)

print("\n=== MODEL SUMMARY ===")
print(df_compare.to_string(index=False))
