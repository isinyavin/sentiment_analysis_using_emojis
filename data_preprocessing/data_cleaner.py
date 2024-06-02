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

single_exclude_list = ['\U0001F171','\U00002122',"\U0001F441",'\U0001F444'] 
double_include_list = [r'[\U0001F1E6-\U0001F1FF]{2}']
triple_include_list = ['\U0001F3F3\U0000FE0F\U0000200D\U0001F308','\U0001F3F4\U0000200D\U00002620\U0000FE0F','\U00002620\U0000FE0F','\U00002764\U0000FE0F',"\U0000263A\U0000FE0F","\U00002639\U0000FE0F", "\U00002665\U0000FE0F","\U0000271D\U0000FE0F","\U0000270C\U0000FE0F","\U00002600\U0000FE0F","\U0000203C\U0000FE0F","\U00002B06\U0000FE0F","\U0000261D\U0000FE0F"]

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


file_path = 'combined.csv'
df = pd.read_csv(file_path)

df.columns = df.columns.str.strip()

print("Trimmed column names in the DataFrame:", df.columns)

def contains_excluded_emoji(text, exclude_list):
    return any(exclude in text for exclude in exclude_list)

df = df[~df['comment_body'].apply(lambda x: contains_excluded_emoji(x, single_exclude_list))]

def get_distinct_emojis(text):
    if isinstance(text, str):
        emoji_list = [match["emoji"] for match in emoji.emoji_list(text)]
        return ''.join(set(emoji_list))
    return ''

def remove_emojis_1str(text):
    if isinstance(text, str):
        return emoji.replace_emoji(text, replace='')
    return text

if 'comment_body' in df.columns:
    df['emojis'] = df['comment_body'].apply(get_distinct_emojis)
    df['cleaned_text'] = df['comment_body'].apply(remove_emojis_1str)
    df = df[(df['cleaned_text'].str.len() > 15) & (df['cleaned_text'].str.len() <= 200)]
    df['cleaned_text'] = df['cleaned_text'].apply(clean_text)

    def extract_emojis(emoji_str):
        if len(emoji_str) > 0:
            combined_pattern = '|'.join(double_include_list + triple_include_list)
            flag_pattern = re.compile(combined_pattern)
            match = flag_pattern.match(emoji_str)
            if match:
                return match.group(0) 
            return emoji_str[0] 
        return ''


    df['emojis'] = df['emojis'].apply(extract_emojis)

    output_path = 'cleaned_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
else:
    print("Error: 'comment_body' column not found in the DataFrame.")

