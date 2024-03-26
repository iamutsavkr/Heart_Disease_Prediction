# Necessary imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load your dataset
df = pd.read_csv('heart.csv')  # replace "your_dataset.csv" with the path to your dataset

# Splitting data into features and target
X = df.drop(columns=['target'])
y = df['target']

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dictionary to store accuracies of different algorithms
accuracies = {}

# Logistic Regression
lr = LogisticRegression(max_iter=1000)  # Increase max_iter to 1000 (or adjust as needed)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
accuracies['Logistic Regression'] = lr_accuracy


# SVM
svm = SVC()
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)
accuracies['SVM'] = svm_accuracy

# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, y_train)
sgd_pred = sgd.predict(X_test)
sgd_accuracy = accuracy_score(y_test, sgd_pred)
accuracies['Stochastic Gradient Descent'] = sgd_accuracy

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
accuracies['Decision Tree'] = dt_accuracy

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
accuracies['Random Forest'] = rf_accuracy

# Finding the best model
best_model_name = max(accuracies, key=accuracies.get)
best_model = None
if best_model_name == 'Logistic Regression':
    best_model = lr
elif best_model_name == 'SVM':
    best_model = svm
elif best_model_name == 'Stochastic Gradient Descent':
    best_model = sgd
elif best_model_name == 'Decision Tree':
    best_model = dt
elif best_model_name == 'Random Forest':
    best_model = rf

best_accuracy = accuracies[best_model_name]

print(f"The best model is {best_model_name} with an accuracy of {best_accuracy}")

# Save the best model to a file
if best_model:
    joblib.dump(best_model, 'best_model.pkl')  # Save the best model for future use
else:
    print("Error: Best model not found!")
