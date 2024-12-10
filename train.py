import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import joblib
import numpy as np

data = pd.read_csv("posture8.csv")
print(data.head())

X = data.drop(["Label"], axis=1)
y = data["Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_accuracy = cross_val_score(model, X, y, cv=kf, scoring="accuracy")
cv_f1 = cross_val_score(model, X, y, cv=kf, scoring="f1_weighted")

print("K-Fold Cross-Validation Results:")
for i in range(len(cv_accuracy)):
    print(f"Fold {i+1}:")
    print(f"   Accuracy: {cv_accuracy[i]:.4f}")
    print(f"   F1 Score: {cv_f1[i]:.4f}")

print("\nMean Accuracy: ", cv_accuracy.mean())
print("Mean F1 Score: ", cv_f1.mean())

plt.figure(figsize=(12, 6))
labels = ["Accuracy", "F1 Score"]
scores = [accuracy, f1]

plt.bar(labels, scores, color=["blue", "orange"])
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Model Performance Metrics")

mean_cv_accuracy = cv_accuracy.mean()
mean_cv_f1 = cv_f1.mean()
plt.bar(["Mean K-Fold Accuracy", "Mean K-Fold F1 Score"], [mean_cv_accuracy, mean_cv_f1], color=["green", "red"])

plt.savefig("model_performance_metrics.png")
plt.close()

plt.figure(figsize=(20, 10))
plot_tree(model.estimators_[0], feature_names=X.columns, filled=True, rounded=True, class_names=["Normal", "Abnormal"])
plt.title("Visualizing Random Forest")
plt.savefig("random_forest_tree.png")
plt.close()

n_trees = len(model.estimators_)
for i in range(n_trees):
    tree = model.estimators_[i]
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    value = tree.tree_.value

    for node in range(n_nodes):
        if children_left[node] != children_right[node]:
            continue
        predicted_class = value[node].argmax()
        print(f"Tree {i}, Node {node}: Predicted class = {predicted_class} (0: Normal, 1: Abnormal)")

# joblib.dump(model, "random_forest_model12.pkl")
print("Model saved as random_forest_model.pkl")
