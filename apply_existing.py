import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np


df = pd.read_csv('output_file.csv')


df['cleaned_text'] = df['cleaned_text'].astype(str)  # Ensure all text entries are strings
df['cleaned_text'] = df['cleaned_text'].fillna('')  # Replace NaNs with empty strings

df['emojis'] = df['emojis'].astype(str)  # Ensure all text entries are strings
df['emojis'] = df['emojis'].fillna('')  # Replace NaNs with empty strings

text_tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
text_tokenizer.fit_on_texts(df['cleaned_text'])
text_sequences = text_tokenizer.texts_to_sequences(df['cleaned_text'])
text_padded = pad_sequences(text_sequences, padding='post')


emoji_tokenizer = Tokenizer(num_words=300, filters='')
emoji_tokenizer.fit_on_texts(df['emojis'])
emoji_counts = pd.Series(emoji_tokenizer.word_counts).sort_values(ascending=False)
top_300_emojis = emoji_counts.head(300)

pd.set_option('display.max_rows', 300)
# Print the 300 most popular emojis and their counts
print("Top 300 emojis and their counts:")
print(top_300_emojis)


model = load_model('emoji_prediction_model.h5')

evaluation_texts = ["that's so great"]
evaluation_sequences = text_tokenizer.texts_to_sequences(evaluation_texts)
evaluation_padded = pad_sequences(evaluation_sequences, maxlen=text_padded.shape[1], padding='post')

predictions = model.predict(evaluation_padded)

for text, prediction in zip(evaluation_texts, predictions):
    top_n = 10
    predicted_indices = np.argsort(prediction)[-top_n:][::-1]
    predicted_emojis = [emoji_tokenizer.index_word[idx] for idx in predicted_indices if idx != 0]
    print(f"Text: {text} -> Predicted Emojis: {predicted_emojis} with Probabilities: {[prediction[idx] for idx in predicted_indices if idx != 0]}")
