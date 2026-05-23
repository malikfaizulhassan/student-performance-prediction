# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
file_path = "student_data.csv"
df = pd.read_csv(file_path)

# Display first rows
print("\nDataset Preview:\n")
print(df.head())

# Features and target
X = df[["StudyHours", "Attendance", "Assignments", "PreviousMarks"]]
y = df["FinalMarks"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation")
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))

# Compare actual vs predicted
results = pd.DataFrame({6
    "Actual": y_test,
    "Predicted": predictions
})

print("\nPredictions:\n")
print(results)

# Visualization
plt.figure(figsize=(8, 5))
plt.scatter(y_test, predictions)
plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("Actual vs Predicted Student Marks")
plt.grid(True)
plt.show()

# Heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# Predict new student
print("\nPredict New Student Performance")

study_hours = float(input("Enter Study Hours: "))
attendance = float(input("Enter Attendance (%): "))
assignments = float(input("Enter Assignment Score: "))
previous_marks = float(input("Enter Previous Marks: "))

new_data = [[study_hours, attendance, assignments, previous_marks]]

predicted_marks = model.predict(new_data)

print("\nPredicted Final Marks:", round(predicted_marks[0], 2))
