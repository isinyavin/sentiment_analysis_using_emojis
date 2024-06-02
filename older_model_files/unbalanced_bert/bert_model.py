import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from transformers import BertTokenizer, TFBertForSequenceClassification, create_optimizer
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import to_categorical
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.callbacks import Callback
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import re

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("Using GPU")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    print("Using CPU")


df = pd.read_csv('data_preprocessing/cleaned_data.csv')
df['cleaned_text'] = df['cleaned_text'].astype(str).fillna('')

df['emojis'] = df['emojis'].astype(str).fillna('')


emoji_tokenizer = Tokenizer(num_words=600, filters='')
emoji_tokenizer.fit_on_texts(df['emojis'])
emoji_sequences = emoji_tokenizer.texts_to_sequences(df['emojis'])
emoji_padded = pad_sequences(emoji_sequences, padding='post')
emoji_labels = np.argmax(to_categorical(emoji_padded, num_classes=len(emoji_tokenizer.word_index) + 1), axis=1)


class_weights = compute_class_weight('balanced', classes=np.unique(emoji_labels), y=emoji_labels)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}


bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

encoded_inputs = bert_tokenizer(df['cleaned_text'].tolist(), padding=True, truncation=True, return_tensors='tf', max_length=128)

train_dataset = tf.data.Dataset.from_tensor_slices((dict(encoded_inputs), emoji_labels)).shuffle(100).batch(32)


model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(emoji_tokenizer.word_index) + 1)


batch_size = 32
num_train_steps = len(df) // batch_size * 3  
optimizer, schedule = create_optimizer(
    init_lr=3e-5,
    num_warmup_steps=0,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01
)


model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])


model.fit(train_dataset, epochs=3, class_weight=class_weights_dict, verbose = 1)


model.save_pretrained('emoji_prediction_model')
