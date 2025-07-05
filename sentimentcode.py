import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon (only once)
nltk.download('vader_lexicon')

# Load the Excel file
file_path = 'ProductReviews2.xlsx'
df = pd.read_excel(file_path)

# Drop missing or empty descriptions
df = df.dropna(subset=['description'])
df = df[df['description'].str.strip() != '']

# Initialize the sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Function to classify sentiment
def get_sentiment(text):
    scores = sid.polarity_scores(str(text))
    compound = scores['compound']
    if compound >= 0.05:
        return 'Positive'
    elif compound <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment analysis
df['Sentiment'] = df['description'].apply(get_sentiment)

# Show sentiment distribution
print("\n📊 Sentiment Distribution:")
print(df['Sentiment'].value_counts())

# Show examples
print("\n🔹 Example Negative Reviews:")
print(df[df['Sentiment'] == 'Negative']['description'].head(5).to_string(index=False))

print("\n🔹 Example Neutral Reviews:")
print(df[df['Sentiment'] == 'Neutral']['description'].head(5).to_string(index=False))

print("\n🔹 Example Positive Reviews:")
print(df[df['Sentiment'] == 'Positive']['description'].head(5).to_string(index=False))
