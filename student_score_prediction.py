"""Predict a sample score from study hours and attendance."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def main():
    csv_path = Path(__file__).with_name("student_scores.csv")
    df = pd.read_csv(csv_path)
    X = df[["StudyHours", "AttendancePercent"]]
    y = df["Score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("Student Score Prediction (sample data)")
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, predictions):.2f}")
    print(f"R-squared: {r2_score(y_test, predictions):.2f}")
    example = pd.DataFrame([{"StudyHours": 5, "AttendancePercent": 80}])
    print(f"Example prediction for 5 study hours and 80% attendance: {model.predict(example)[0]:.1f}")
    print("\nNote: This is demonstration data and is not suitable for real student decisions.")

if __name__ == "__main__":
    main()
