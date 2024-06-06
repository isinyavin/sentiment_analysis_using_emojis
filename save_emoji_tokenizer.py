import json
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tensorflow.keras.utils import to_categorical
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import re

df = pd.read_csv('data_preprocessing/cleaned_data.csv')
df['cleaned_text'] = df['cleaned_text'].astype(str).fillna('')

df['emojis'] = df['emojis'].astype(str).fillna('')


emoji_tokenizer = Tokenizer(num_words=650, filters='')
emoji_tokenizer.fit_on_texts(df['emojis'])
emoji_sequences = emoji_tokenizer.texts_to_sequences(df['emojis'])
emoji_padded = pad_sequences(emoji_sequences, padding='post')
emoji_labels = np.argmax(to_categorical(emoji_padded, num_classes=len(emoji_tokenizer.word_index) + 1), axis=1)


def save_emoji_tokenizer(tokenizer, file_path):
    tokenizer_config = {
        'word_index': tokenizer.word_index
    }
    with open(file_path, 'w') as file:
        json.dump(tokenizer_config, file)


save_emoji_tokenizer(emoji_tokenizer, 'emoji_tokenizer_config2.json')

