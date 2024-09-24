import pandas as pd

# Read the first Excel file
df1 = pd.read_excel("C:\\Users\\Rama Krishna\\Desktop\\17 MAXCUT\\17thc1tweetstsp.xlsx")

# Read the second Excel file
df2 = pd.read_excel("C:\\Users\\Rama Krishna\\Desktop\\17 MAXCUT\\17thc2tweetstsp.xlsx")

# Compute total sentiment scores for each DataFrame
total_score_df1 = df1['Total Sentiment Score'].sum()
total_score_df2 = df2['Total Sentiment Score'].sum()

print("Total Sentiment Score for DataFrame 1:", total_score_df1)
print("Total Sentiment Score for DataFrame 2:", total_score_df2)

# Update stance based on total sentiment scores
if total_score_df1 > total_score_df2:
    df1['Stance'] = 1
    df2['Stance'] = 0
else:
    df1['Stance'] = 0
    df2['Stance'] = 1

# Save the updated DataFrames to new Excel files
df1.to_excel("C:\\Users\\Rama Krishna\\Desktop\\17 MAXCUT\\17thc1tweetswithtspwithstance.xlsx", index=False)
df2.to_excel("C:\\Users\\Rama Krishna\\Desktop\\17 MAXCUT\\17thc2tweetswithtspwithstance.xlsx",index=False)