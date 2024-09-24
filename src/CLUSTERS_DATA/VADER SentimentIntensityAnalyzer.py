import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download NLTK resources
nltk.download('vader_lexicon')

# Initialize the VADER sentiment intensity analyzer
sid = SentimentIntensityAnalyzer()

# Function to calculate sentiment metrics for a given text
def calculate_sentiment_metrics(text):
    # Get the polarity scores for the text
    sentiment_scores = sid.polarity_scores(text)
    
    # Positive and negative count
    positive_count = sum(1 for score in sentiment_scores.values() if score > 0)
    negative_count = sum(1 for score in sentiment_scores.values() if score < 0)
    
    # Average positive and negative score
    avg_positive_score = sum(score for score in sentiment_scores.values() if score > 0) / positive_count if positive_count > 0 else 0
    avg_negative_score = sum(score for score in sentiment_scores.values() if score < 0) / negative_count if negative_count > 0 else 0
    
    # Total sentiment score
    total_score = sentiment_scores['compound']
    
    # Positive sentiment score
    positive_score = sentiment_scores['pos']
    
    # Negative sentiment score
    negative_score = sentiment_scores['neg']
    
    # Return sentiment metrics as a dictionary
    return {
        'Positive Count': positive_count,
        'Negative Count': negative_count,
        'Avg Positive': avg_positive_score,
        'Avg Negative': avg_negative_score,
        'Total Sentiment Score': total_score,
        'Positive Sentiment Score': positive_score,
        'Negative Sentiment Score': negative_score
    }

# Read the Excel file
df = pd.read_excel("C:\\Users\\Rama Krishna\\Desktop\\TextData.xlsx")

# Apply the function to calculate sentiment metrics for each tweet
sentiment_metrics = df['Text'].apply(calculate_sentiment_metrics)

# Convert the list of dictionaries to a DataFrame
sentiment_metrics_df = pd.DataFrame(sentiment_metrics.tolist())

# Concatenate the original DataFrame with the new sentiment metrics DataFrame
df = pd.concat([df, sentiment_metrics_df], axis=1)

# Save the DataFrame with sentiment metrics to a new Excel file
df.to_excel("C:\\Users\\Rama Krishna\\Documents\\rk865.xlsx", index=False)
