import pandas as pd

csv_files = ["reddit_comments_with_emojis2.csv", "reddit_comments_with_emojis3.csv", "reddit_comments_with_emojis4.csv","reddit_comments_with_emojis5.csv","reddit_comments_with_emojis6.csv","reddit_comments_with_emojis7.csv","reddit_comments_with_emojis8.csv","reddit_comments_with_emojis9.csv","reddit_comments_with_emojis10.csv","reddit_comments_with_emojis11.csv", "reddit_comments_with_emojis12.csv","reddit_comments_with_emojis13.csv","reddit_comments_with_emojis14.csv","reddit_comments_with_emojis15.csv","reddit_comments_with_emojis16.csv","reddit_comments_with_emojis17.csv","reddit_comments_with_emojis18.csv"]

dataframes = [pd.read_csv(file) for file in csv_files]

combined_df = pd.concat(dataframes, ignore_index=True)

combined_df.to_csv("combined.csv", index=False)

print("CSV files have been combined into combined.csv")
