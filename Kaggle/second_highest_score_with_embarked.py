##this simple model training with 5 columns Pclass, Sex,Sibsp, Parch, and Fare, and using Random Forest Classifier
##achieved 0.79186 score on Kaggle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import randint

# Select features (now includes Embarked)
features = ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Embarked"]

# Convert 'Sex' to numerical (Female=0, Male=1)
le_sex = LabelEncoder()
train_data["Sex"] = le_sex.fit_transform(train_data["Sex"])
test_data["Sex"] = le_sex.transform(test_data["Sex"])

# Convert 'Embarked' to numerical
le_embarked = LabelEncoder()
train_data["Embarked"] = le_embarked.fit_transform(train_data["Embarked"].astype(str))
test_data["Embarked"] = le_embarked.transform(test_data["Embarked"].astype(str))

# Fill missing Fare values (if any) with the mean
train_data["Fare"] = train_data["Fare"].fillna(train_data["Fare"].mean())
test_data["Fare"] = test_data["Fare"].fillna(test_data["Fare"].mean())

# Normalize Fare
scaler = StandardScaler()
train_data[["Fare"]] = scaler.fit_transform(train_data[["Fare"]])
test_data[["Fare"]] = scaler.transform(test_data[["Fare"]])

# Prepare training and testing datasets
X = train_data[features]
y = train_data["Survived"]
X_test = test_data[features]

# Hyperparameter tuning with RandomizedSearchCV
param_dist = {
    "n_estimators": randint(50, 300),
    "max_depth": [3, 5, 10, None],
    "min_samples_split": randint(2, 10),
    "min_samples_leaf": randint(1, 5),
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=1), param_distributions=param_dist, 
    n_iter=20, cv=5, scoring="accuracy", random_state=1
)

# Train the optimized model
random_search.fit(X, y)
best_model = random_search.best_estimator_

# Make predictions
predictions = best_model.predict(X_test)

# Save the results
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)

# Print results
print("Predictions:", predictions)
print("Submission file saved successfully!")

#adding embarked brought the score 0.76