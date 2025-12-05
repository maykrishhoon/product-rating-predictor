import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

print("\n--- RATING PREDICTOR MODEL TRAINING ---\n")

# Sample dataset creation

data = {
    'review': [
        "This product is amazing, I love it!",
        "Terrible, waste of money.",
        "It's okay, not the best but works.",
        "Excellent quality and fast shipping.",
        "Broken upon arrival, very sad.",
        "Good value for the price.",
        "I hate this, worst purchase ever.",
        "Five stars, absolutely perfect.",
        "Mediocre performance, expected better.",
        "Superb! Highly recommended.",
        "Does not work as advertised.",
        "Pretty good, but battery life is short.",
        "Awful experience, customer service was rude.",
        "Fantastic design and intuitive use.",
        "Just average, nothing special."
    ],
    'rating': [5, 1, 3, 5, 1, 4, 1, 5, 2, 5, 1, 3, 1, 5, 3]
}

df = pd.DataFrame(data)

print("--- Sample Data ---")
print(df.head())
print("\n")

vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)

X = vectorizer.fit_transform(df['review'])
y = df['rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("Training complete! Model has learned from the data.")

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse:.2f} (Lower is better)")
print("-" * 30)


def predict_rating(new_review):
    
    new_review_vector = vectorizer.transform([new_review])
    
    prediction = model.predict(new_review_vector)
    
    rating = np.clip(prediction[0], 1, 5)
    return rating


test_sentences = [
    "I absolutely loved this product, it is great!",
    "Total garbage, do not buy.",
    "It is okay, maybe a bit expensive."
]

print("\n--- LIVE PREDICTIONS ---")
for sentence in test_sentences:
    rating = predict_rating(sentence)
    print(f"Review: '{sentence}'")
    print(f"Predicted Rating: {rating:.1f} / 5.0")
    print("-" * 20)
