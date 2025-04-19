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
import cv2

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# not printing because there are too many and takes forever 

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# first 4 lines of output
# /kaggle/input/histopathologic-cancer-detection/sample_submission.csv
# /kaggle/input/histopathologic-cancer-detection/train_labels.csv
# /kaggle/input/histopathologic-cancer-detection/test/a7ea26360815d8492433b14cd8318607bcf99d9e.tif
# /kaggle/input/histopathologic-cancer-detection/test/59d21133c845dff1ebc7a0c7cf40c145ea9e9664.tif
# -----------------------------------------------------------------------------------
# Load dataset
file_path = '/kaggle/input/histopathologic-cancer-detection/train_labels.csv'
df = pd.read_csv(file_path)

# Display basic information
print("Dataset Shape:", df.shape)
print("\nColumn Data Types:\n", df.dtypes)
# -----------------------------------------------------------------------------------
# Check for missing values
missing_values = df.isnull().sum()
# Display the missing values
print("\nMissing Values:\n", df.isnull().sum())

# -----------------------------------------------------------------------------------
# Display all columns
pd.set_option('display.max_columns', None)

# Display first few rows
display(df.head())

# Summary statistics
display(df.describe())

print(df['label'].value_counts())
# -----------------------------------------------------------------------------------
# Plot histogram for Histopathologic Cancer Detection
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='label')
plt.title("Cancer detection distribution")
plt.xlabel("label")
plt.ylabel("id")
plt.show()
# -----------------------------------------------------------------------------------
# Random Forest Classifier

# Load labels (assuming you have train_labels.csv)
labels = pd.read_csv('/kaggle/input/histopathologic-cancer-detection/train_labels.csv')

# Extract features from images (simplified example)
def extract_features(img_ids, base_path):
    features = []
    for img_id in img_ids:
        img = cv2.imread(f"{base_path}/{img_id}.tif")
        img = cv2.resize(img, (32, 32))  # Reduce size for faster processing
        features.append(img.flatten())  # Flatten to 1D array (3072 features for 32x32x3)
    return np.array(features)

# Use a SUBSET for testing (remove [0:1000] for full dataset)
train_features = extract_features(labels['id'][0:1000], 
                                '/kaggle/input/histopathologic-cancer-detection/train')
# -----------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    train_features, 
    labels['label'][0:1000],  # Match subset size
    test_size=0.2,
    random_state=42
)
# -----------------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=50,  # Reduced for faster testing
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
rf.fit(X_train, y_train)
# -----------------------------------------------------------------------------------
from sklearn.metrics import classification_report

y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))
# -----------------------------------------------------------------------------------

