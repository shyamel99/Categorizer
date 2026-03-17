from .utils import load_model

def predict_text(text: str):
    model, vectorizer = load_model("models/text_classifier.pkl")
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    return prediction

def main():
    print("Text Classifier — Prediction Mode")
    print("----------------------------------")
    
    while True:
        user_input = input("\nEnter a message (or type 'exit'): ")

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        label = predict_text(user_input)
        print(f"Predicted label: {label}")

if __name__ == "__main__":
    main()
