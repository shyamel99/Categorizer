import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

from .utils import (
    load_data,
    train_test_split_data,
    create_vectorizer,
    save_model,
)


def main():
    # 1. Load data
    print("Loading data...")
    df = load_data("data/generated_1000_support_tickets.csv")
    print(f"Loaded {len(df)} rows.")

    # 2. Split into train and test
    print("Splitting into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split_data(df)

    # 3. Create and fit the vectorizer
    print("Creating and fitting TF-IDF vectorizer...")
    vectorizer = create_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 4. Create and train the classifier
    print("Training classifier...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)

    # 5. Evaluate on the test set
    print("Evaluating on test set...")
    y_pred = clf.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy on test set: {acc:.2f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    # 6. Save the trained model and vectorizer
    model_path = os.path.join("models", "text_classifier.pkl")
    print(f"\nSaving model to {model_path} ...")
    save_model(clf, vectorizer, model_path)
    print("Model saved successfully.")


if __name__ == "__main__":
    main()
