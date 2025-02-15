import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

# Load the dataset
train_data = pd.read_csv('data/pendigits_dyn_train.csv', header=None)
test_data = pd.read_csv('data/pendigits_dyn_test.csv', header=None)
train_labels = pd.read_csv('data/pendigits_label_train.csv', header=None)
test_labels = pd.read_csv('data/pendigits_label_test.csv', header=None)

# Combine train and test data
X = np.vstack([train_data.values, test_data.values])
y_orig = np.concatenate([train_labels.values.ravel(), test_labels.values.ravel()])

# Train Isolation Forest
iso_forest = IsolationForest(random_state=42, contamination=0.1)
y_scores = -iso_forest.fit_predict(X)

# Calculate and print AUC for each class
print("AUC Scores when treating each class as anomalous:")
for unique_class in np.unique(y_orig):
    # Create binary labels (1 for normal, -1 for anomaly)
    y_binary = np.where(y_orig == unique_class, -1, 1)
    # Calculate anomaly ratio for this class
    anomaly_ratio = (y_binary == -1).sum() / len(y_binary)
    # Calculate AUC
    auc = roc_auc_score(y_binary == -1, y_scores)
    print(f"Class {unique_class}: AUC = {auc:.3f} (represents {anomaly_ratio:.3%} of the data)")