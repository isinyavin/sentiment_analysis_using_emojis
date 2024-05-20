import pandas as pd
import emoji
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    
    text = re.sub(r"n't", ' not', text)
    text = re.sub(r"'re", ' are', text)
    text = re.sub(r"'s", ' is', text)
    text = re.sub(r"'d", ' would', text)
    text = re.sub(r"'ll", ' will', text)
    text = re.sub(r"'t", ' not', text)
    text = re.sub(r"'ve", ' have', text)
    text = re.sub(r"'m", ' am', text)


    text = re.sub(r'[^a-zA-Z!? ]+', '', text)

    words = nltk.word_tokenize(text)

    #words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    cleaned_text = ' '.join(words)
    
    return cleaned_text


#reads one of the reddit csv files
file_path = 'combined.csv'
df = pd.read_csv(file_path)

df.columns = df.columns.str.strip()

print("Trimmed column names in the DataFrame:", df.columns)

#returns the distinct emojis in each comment body
def get_distinct_emojis(text):
    if isinstance(text, str):
        emoji_list = [match["emoji"] for match in emoji.emoji_list(text)]
        return ''.join(set(emoji_list))
    return ''

#function that removes emojis from text
def remove_emojis_1str(text):
    if isinstance(text, str):
        return emoji.replace_emoji(text, replace='')
    return text

#extracts texts and emojis from the full comment body
if 'comment_body' in df.columns:
    df['emojis'] = df['comment_body'].apply(get_distinct_emojis)
    df['cleaned_text'] = df['comment_body'].apply(remove_emojis_1str)
    #selects comment bodies that are between 1 and 200 characters
    df = df[(df['cleaned_text'].str.len() > 1) & (df['cleaned_text'].str.len() <= 200)]
    df['cleaned_text'] = df['cleaned_text'].apply(clean_text)
    # if comment bodies have more more than 1 distinct emojis, then this will create x amount of rows with x distinct emojis. if more than one of the same emoji, then this displays 1 row with 1 emoji.
    # Choose one emoji (the first one) from the distinct set
    df['emojis'] = df['emojis'].apply(lambda x: x[0] if len(x) > 0 else '')

    output_path = 'output_file.csv'
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
else:
    print("Error: 'comment_body' column not found in the DataFrame.")

