import pandas as pd
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

data = {'cleaned_text': ["Hello! How are you?", "I'm fine. Thanks!"]}
df = pd.DataFrame(data)

def separate_punctuation(text):
    text = re.sub(r'([!?.])', r' \1', text)
    return text

df['cleaned_text'] = df['cleaned_text'].apply(separate_punctuation)
#print(df['cleaned_text'])

custom_filters = '\"#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n'
text_tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>", filters=custom_filters)
text_tokenizer.fit_on_texts(df['cleaned_text'])
text_sequences = text_tokenizer.texts_to_sequences(df['cleaned_text'])
text_padded = pad_sequences(text_sequences, padding='post')

word_index = text_tokenizer.word_index
print(word_index)

print(text_padded)