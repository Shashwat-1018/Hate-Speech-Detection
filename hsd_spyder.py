
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import joblib

loaded_model = joblib.load("C:/Users/shash/Desktop/Hate Speech Detection/hsd_model.joblib")

positive_words = {"love", "nice", "good", "happy", "great", "friend", "beautiful"}

def custom_predict(text):
    base_pred = loaded_model.predict([text])[0]
    proba = loaded_model.predict_proba([text])[0]

    # Positive safeguard
    if any(word in text.lower() for word in positive_words):
        not_off_idx = list(loaded_model.classes_).index("Not Offensive")
        if proba[not_off_idx] < 0.6:
            base_pred = "Not Offensive"

    return base_pred, dict(zip(loaded_model.classes_, proba))

print("Enter a phrase to check if it's hate speech (type 'quit' to exit):")
while True:
    user_input = input("Enter text: ")
    if user_input.lower() == 'quit':
        break
    pred, probas = custom_predict(user_input)
    print(f"\nPrediction: {pred}")
    print("Confidence Scores:")
    for label, prob in probas.items():
        print(f"  {label}: {prob:.4f}")
