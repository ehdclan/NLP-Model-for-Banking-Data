import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    df = pd.read_csv('bank_intents_new.csv')

    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )

    with open('bank_model.pkl', 'rb') as f:
        vectorizer, model = pickle.load(f)

    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2f}\n")

    print("Classification report:")
    print(classification_report(y_test, y_pred))

    # print("Confusion matrix:")
    # print(confusion_matrix(y_test, y_pred))


if __name__ == '__main__':
    main()
