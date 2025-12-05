# product-rating-predictor
E-commerce platforms receive thousands of reviews daily. Manually reading them to assign a star rating is impossible. The goal of this project was to build an AI that can read text and predict a numeric rating (1-5).


🗣️ Product Review Analysis (NLP)

A Natural Language Processing (NLP) project that analyzes customer reviews to determine sentiment and extract key topics.

🎯 Project Goal

To automate the process of reading reviews. Instead of manually reading 1,000 comments, this tool automatically:

Classifies Sentiment: Tells us if a review is Positive, Negative, or Neutral.

Extracts Keywords: Identifies what people are talking about (e.g., "battery", "screen", "price").

🛠️ Tech Stack

Python

TextBlob: For calculating sentiment polarity (identifying emotions).

Scikit-Learn: Used CountVectorizer to extract keywords and remove stop-words.

Pandas & Matplotlib: For data management and visualization.

🧠 How It Works

Input: Raw text reviews.

Sentiment Engine: TextBlob scans the text for adjectives.

"Great", "Love" → Positive Score (+1)

"Terrible", "Broken" → Negative Score (-1)

Keyword Extraction: The script removes common words ("the", "is") and counts the remaining nouns/verbs to find common complaints or praises.

📊 Sample Output

Review

Sentiment

"The battery life is terrible."

Negative

"I absolutely love this phone!"

Positive

Top Keywords: battery, price, quality.

🚀 How to Run

Clone the repository.

Install libraries:

pip install pandas matplotlib textblob scikit-learn


Run the script:

rating_pridictor.py


Created by KRISH as part of the Indikraft Internship.
