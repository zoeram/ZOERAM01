

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
train_data = pd.read_csv(r'E:\Dream\machine learning\connect-four\c4-train.txt', header=None)
test_data = pd.read_csv(r'E:\Dream\machine learning\connect-four\c4-test.txt', header=None)
validation_data = pd.read_csv(r'E:\Dream\machine learning\connect-four\c4-validation.txt', header=None)

from sklearn.preprocessing import OneHotEncoder

# Function to preprocess the data
def preprocess_data(data):
    X = data.apply(lambda row: list(row[0]), axis=1)  # Convert each row to a list of characters
    X_encoded = []
    for row in X:
        encoded_row = []
        for char in row:
            if char == '.':
                encoded_row.extend([0, 0, 0])  # Encode '.' as [0, 0, 0]
            elif char == 'Y':
                encoded_row.extend([1, 0, 0])  # Encode 'Y' as [1, 0, 0]
            elif char == 'R':
                encoded_row.extend([0, 1, 0])  # Encode 'R' as [0, 1, 0]
            else:
                encoded_row.extend([0, 0, 1])  # Encode other characters as [0, 0, 1]
        X_encoded.append(encoded_row)
    y = data.iloc[:, -1].apply(lambda x: x[-1]).values  # Extract the last character as the target variable

    return X_encoded, y

# Preprocess the training, test, and validation data
X_train, y_train = preprocess_data(train_data)
X_test, y_test = preprocess_data(test_data)
X_validation, y_validation = preprocess_data(validation_data)

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Initialize the GridSearchCV object
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=5,  # 5-fold cross-validation
                           scoring='accuracy',  # Use accuracy as the scoring metric
                           n_jobs=-1)  # Use all available CPU cores

# Perform the grid search
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Use the best model for prediction
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate the best model
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# visualize the distribution of target variable 
plt.figure(figsize=(8, 6))
pd.Series(y_train).value_counts().plot(kind='bar')
plt.title('Distribution of Target Variable')
plt.xlabel('Target Variable')
plt.ylabel('Count')
plt.show()
