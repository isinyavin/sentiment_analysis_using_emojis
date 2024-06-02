import pandas as pd

csv_files = ["reddit_comments_with_emojis2.csv", "reddit_comments_with_emojis3.csv", "reddit_comments_with_emojis4.csv","reddit_comments_with_emojis5.csv","reddit_comments_with_emojis6.csv","reddit_comments_with_emojis7.csv","reddit_comments_with_emojis8.csv","reddit_comments_with_emojis9.csv","reddit_comments_with_emojis10.csv","reddit_comments_with_emojis11.csv", "reddit_comments_with_emojis12.csv","reddit_comments_with_emojis13.csv","reddit_comments_with_emojis14.csv","reddit_comments_with_emojis15.csv","reddit_comments_with_emojis16.csv","reddit_comments_with_emojis17.csv","reddit_comments_with_emojis18.csv","reddit_comments_with_emojis19.csv","reddit_comments_with_emojis20.csv","reddit_comments_with_emojis21.csv","reddit_comments_with_emojis22.csv","reddit_comments_with_emojis23.csv","reddit_comments_with_emojis24.csv","reddit_comments_with_emojis25.csv", "reddit_comments_with_emojis26.csv","reddit_comments_with_emojis27.csv","reddit_comments_with_emojis28.csv","reddit_comments_with_emojis29.csv","reddit_comments_with_emojis30.csv","reddit_comments_with_emojis31.csv","reddit_comments_with_emojis32.csv","reddit_comments_with_emojis33.csv","reddit_comments_with_emojis34.csv","reddit_comments_with_emojis35.csv","reddit_comments_with_emojis36.csv","reddit_comments_with_emojis37.csv","reddit_comments_with_emojis38.csv","reddit_comments_with_emojis39.csv","reddit_comments_with_emojis40.csv","reddit_comments_with_emojis41.csv","reddit_comments_with_emojis42.csv","reddit_comments_with_emojis43.csv","reddit_comments_with_emojis44.csv","reddit_comments_with_emojis45.csv","reddit_comments_with_emojis46.csv","reddit_comments_with_emojis47.csv","reddit_comments_with_emojis48.csv","reddit_comments_with_emojis49.csv"]

dataframes = [pd.read_csv(file) for file in csv_files]

combined_df = pd.concat(dataframes, ignore_index=True)

combined_df.to_csv("combined.csv", index=False)

print("CSV files have been combined into combined.csv")
