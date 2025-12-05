# product-rating-predictor
E-commerce platforms receive thousands of reviews daily. Manually reading them to assign a star rating is impossible. The goal of this project was to build an AI that can read text and predict a numeric rating (1-5).


🌟 Product Rating Predictor

A Machine Learning project that predicts the star rating (1-5) of a product based purely on the text of the customer review.

📖 Overview

This project uses Natural Language Processing (NLP) to analyze the sentiment of text reviews and a Linear Regression model to predict a numerical rating. It was built to understand how computers interpret human language and convert subjective opinions into quantitative data.

🛠️ Tech Stack

Language: Python 3.9+

Libraries: Pandas, Scikit-Learn, Numpy

Technique: TF-IDF (Term Frequency-Inverse Document Frequency) for vectorization.

⚙️ How It Works

Data Collection: Uses a dataset of product reviews and their associated ratings.

Preprocessing: * Converts text to lowercase.

Removes "stop words" (common words like 'the', 'and', 'is' that don't carry sentiment).

Vectorizes text using TF-IDF.

Training: Fits a Linear Regression model to find the correlation between specific words and high/low ratings.

Prediction: The model accepts new text input and outputs a predicted score (e.g., 4.2 stars).

🚀 How to Run

Clone the repository:

git clone


Install dependencies:

pip install pandas scikit-learn


Run the script:

rating_predictor.py


📊 Example Results

Input Review

Predicted Rating

"I absolutely loved this product!"

4.9 / 5.0

"It is okay, maybe a bit expensive."

3.1 / 5.0

"Total garbage, do not buy."

1.2 / 5.0

🔮 Future Improvements

Implement a Random Forest Regressor for potentially better accuracy.

Train on a larger real-world dataset (like the Amazon Reviews dataset).

Deploy as a simple web app using Streamlit.


Created by KRISH as part of the Indikraft Internship.
