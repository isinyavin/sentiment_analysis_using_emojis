from fastapi import FastAPI, Request
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import re
import pandas as pd
import pandas as pd
import emoji
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


model = tf.keras.models.load_model('emoji_prediction_model.h5')

app = FastAPI()

class TextRequest(BaseModel):
    text: str

df = pd.read_csv('output_file.csv')
df['cleaned_text'] = df['cleaned_text'].astype(str).fillna('')
df['emojis'] = df['emojis'].astype(str).fillna('')

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
    cleaned_text = ' '.join(words)
    return cleaned_text

def separate_punctuation(text):
    text = re.sub(r'([!?.])', r' \1', text)
    return text

df['cleaned_text'] = df['cleaned_text'].apply(separate_punctuation)

custom_filters = '\"#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n'
text_tokenizer = Tokenizer(num_words=1000000, oov_token="<OOV>", filters=custom_filters)
text_tokenizer.fit_on_texts(df['cleaned_text'])

emoji_tokenizer = Tokenizer(num_words=650, filters='')
emoji_tokenizer.fit_on_texts(df['emojis'])

text_padded_shape = pad_sequences(text_tokenizer.texts_to_sequences(df['cleaned_text']), padding='post').shape[1]

@app.post("/predict_emojis")
async def predict_emojis(request: TextRequest):
    text = request.text
    text = clean_text(text)
    sequence = text_tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=text_padded_shape, padding='post')

    prediction = model.predict(padded_sequence)
    top_n = 10
    predicted_indices = np.argsort(prediction[0])[-top_n:][::-1]
    predicted_emojis = [emoji_tokenizer.index_word[idx] for idx in predicted_indices if idx != 0]

    return {"text": text, "predicted_emojis": predicted_emojis, "probabilities": [float(prediction[0][idx]) for idx in predicted_indices if idx != 0]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8011)