"""Iris flower classification using a Decision Tree."""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
    )
    model = DecisionTreeClassifier(random_state=42, max_depth=3)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("Iris Flower Classification")
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=iris.target_names))

if __name__ == "__main__":
    main()
