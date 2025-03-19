import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from sklearn.model_selection import GridSearchCV

# Load the datasets
train_data = pd.read_csv("train_campusrecruit.csv")
test_data = pd.read_csv("test_campusrecruit.csv")

# Identify numerical and categorical columns
numerical_cols = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]
categorical_cols = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation"]

# Fill missing values in numerical columns with median
train_data[numerical_cols] = train_data[numerical_cols].fillna(train_data[numerical_cols].median())
test_data[numerical_cols] = test_data[numerical_cols].fillna(test_data[numerical_cols].median())

# Fill missing values in categorical columns with mode
for col in categorical_cols:
    train_data[col] = train_data[col].fillna(train_data[col].mode()[0])
    test_data[col] = test_data[col].fillna(test_data[col].mode()[0])

# Drop rows with missing target variable
train_data.dropna(subset=["status"], inplace=True)

# Separate features and target for classification (status prediction)
X = train_data.drop(columns=["status", "salary", "id"])
y = train_data["status"]

# One-hot encoding for categorical variables
X_encoded = pd.get_dummies(X, drop_first=True)
test_encoded = pd.get_dummies(test_data.drop(columns=["id"]), drop_first=True)

# Ensure test data has same columns as training data
missing_cols = set(X_encoded.columns) - set(test_encoded.columns)
for col in missing_cols:
    test_encoded[col] = 0  # Add missing columns with 0s
test_encoded = test_encoded[X_encoded.columns]  # Reorder columns to match

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Train the best model
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)
best_model.fit(X_train, y_train)

# Evaluate the model
y_pred = best_model.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))

# Predict on test data
test_predictions = best_model.predict(test_encoded)

# Save classification predictions
submission = pd.DataFrame({"id": test_data["id"], "status": test_predictions})
submission.to_csv("submission.csv", index=False)
print("Classification submission file saved as submission.csv")

# Salary prediction (Regression model only for 'Placed' students)
placed_train = train_data[train_data["status"] == "Placed"].drop(columns=["status", "id"])
placed_train.dropna(subset=["salary"], inplace=True)

X_salary = placed_train.drop(columns=["salary"])
y_salary = placed_train["salary"]

# One-hot encoding for salary prediction
X_salary_encoded = pd.get_dummies(X_salary, drop_first=True)

# Train-test split for salary prediction
X_train_salary, X_val_salary, y_train_salary, y_val_salary = train_test_split(X_salary_encoded, y_salary, test_size=0.2, random_state=42)

# Train RandomForestRegressor for salary prediction
salary_model = RandomForestRegressor(random_state=42)
salary_model.fit(X_train_salary, y_train_salary)

# Evaluate salary model
y_pred_salary = salary_model.predict(X_val_salary)
print("Salary Prediction MAE:", mean_absolute_error(y_val_salary, y_pred_salary))

# Predict salary for test dataset where status is 'Placed'
test_placed = test_data[test_data["id"].isin(submission[submission["status"] == "Placed"]["id"])]
test_placed_encoded = pd.get_dummies(test_placed.drop(columns=["id"]), drop_first=True)
test_placed_encoded = test_placed_encoded[X_salary_encoded.columns]

test_salary_predictions = salary_model.predict(test_placed_encoded)

# Save salary predictions
salary_submission = pd.DataFrame({"id": test_placed["id"], "salary": test_salary_predictions})
salary_submission.to_csv("salary_submission.csv", index=False)
print("Salary submission file saved as salary_submission.csv")
