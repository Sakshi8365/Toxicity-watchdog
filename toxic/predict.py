import joblib

# Load trained model and vectorizer
model = joblib.load('toxic_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

print("🎮 Toxic Chat Detector is ready! Type messages to check (type 'exit' to quit)")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("👋 Exiting the detector. GG!")
        break

    # Convert input text to vector
    vector = vectorizer.transform([user_input])

    # Make prediction
    prediction = model.predict(vector)[0]

    # Show result
    if prediction == 1:
        print("🚨 This message is TOXIC!")
    else:
        print("✅ This message is CLEAN.")
