import pandas as pd

# Load the Excel file
df = pd.read_excel("C:\\Users\\Rama Krishna\\Desktop\\17 MAXCUT\\17thc2tweetsts.xlsx")

# Assuming the sentiment score column is named 'sentiment_score'
# Create a new column 'predicted_stance' based on the condition
df['predicted_stance'] = df['Total Sentiment Score'].apply(lambda x: 1 if x > 0 else 0)

# Save the updated dataframe to a new Excel file
df.to_excel("C:\\Users\\Rama Krishna\\Desktop\\17 MAXCUT\\17thc2tweetstsp.xlsx", index=False)
