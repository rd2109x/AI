# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)

men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived", rate_men)

#this is so that i dont have to run all the others separately. 

# Load dataset
file_path = '/kaggle/input/titanic/train.csv'
df = pd.read_csv(file_path)

# Display basic information
print("Dataset Shape:", df.shape)
print("\nColumn Data Types:\n", df.dtypes)

# Check for missing values
missing_values = df.isnull().sum()
# Display the missing values
print("\nMissing Values:\n", df.isnull().sum())

# Display all columns
pd.set_option('display.max_columns', None)

# Display first few rows
display(df.head())

# Summary statistics
display(df.describe())

# Plot histogram for Passenger Survival
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Survived')
plt.title("Passenger Survival Distribution")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.show()

# Compute correlation matrix
correlation_matrix = df.corr(numeric_only=True)

# Plot correlation heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.show()

df = df.drop(columns = ["Ticket", "Cabin", "Name"] )

df.head()

# Convert string columns to integers
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

df.head()

# Recompute correlation matrix with converted data
correlation_matrix_updated = df.corr(numeric_only=True)

# Plot updated correlation heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix_updated, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Updated Correlation Matrix Heatmap After Encoding")
plt.show()

# One-hot encode remaining categorical columns
#df = pd.get_dummies(df, columns=["Age"])
# Fill NaN values in "Age" with the mean value

# Define features and target
X = df.drop(columns=["Survived"])
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train['Age'] = X_train['Age'].fillna(X_train['Age'].mean())
X_test['Age'] = X_test['Age'].fillna(X_test['Age'].mean())
# Fill NaN values in "Embarked" with the mode
X_train['Embarked'] = X_train['Embarked'].fillna(X_train['Embarked'].mode()[0])
X_test['Embarked'] = X_test['Embarked'].fillna(X_test['Embarked'].mode()[0])


print(X_train)
df.head()

# Check for NaN values in the entire DataFrame
#print(X_train.isna().sum())
#print(X_test.isna().sum())

#df.head()
# Train Decision Tree Classifier
dt_clf = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
dt_clf.fit(X_train, y_train)

# Predict on test set
y_pred_dt = dt_clf.predict(X_test)

# Evaluate the Decision Tree model
print("Decision Tree Classifier Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("\nClassification Report:\n", classification_report(y_test, y_pred_dt))

# Visualize the Decision Tree
plt.figure(figsize=(16, 10))
tree.plot_tree(dt_clf, feature_names=X.columns, class_names=["No Survived", "Survived"], filled=True, fontsize=6)
plt.title("Decision Tree Visualization")
plt.show()

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate model
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Get feature importance scores
importances = rf_model.feature_importances_
feature_names = X.columns

# Plot feature importances
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(8, 5))
plt.title("Feature Importance - Random Forest")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=45)
plt.show()
