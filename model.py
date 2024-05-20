import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.callbacks import Callback
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


df = pd.read_csv('output_file.csv')
df['cleaned_text'] = df['cleaned_text'].astype(str)  # Ensure all text entries are strings
df['cleaned_text'] = df['cleaned_text'].fillna('') 

df['emojis'] = df['emojis'].astype(str)  # Ensure all text entries are strings
df['emojis'] = df['emojis'].fillna('')  # Replace NaNs with empty strings

text_tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
text_tokenizer.fit_on_texts(df['cleaned_text'])
text_sequences = text_tokenizer.texts_to_sequences(df['cleaned_text'])
text_padded = pad_sequences(text_sequences, padding='post')

emoji_tokenizer = Tokenizer(num_words=100, filters='')
emoji_tokenizer.fit_on_texts(df['emojis'])
emoji_sequences = emoji_tokenizer.texts_to_sequences(df['emojis'])
emoji_padded = pad_sequences(emoji_sequences, padding='post')
#print(emoji_padded)

emoji_categorical = tf.keras.utils.to_categorical(emoji_padded, num_classes=len(emoji_tokenizer.word_index) + 1)

# Compute class weights
emoji_labels = np.argmax(emoji_categorical, axis=1)
class_weights = compute_class_weight('balanced', classes=np.unique(emoji_labels), y=emoji_labels)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Define the model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=64, input_length=text_padded.shape[1]))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(len(emoji_tokenizer.word_index) + 1, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Custom callback to print progress
class TrainingProgressCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1} finished. Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}, Val Loss: {logs['val_loss']:.4f}, Val Accuracy: {logs['val_accuracy']:.4f}")

# Train the model with progress updates and class weights
model.fit(text_padded, emoji_categorical, epochs=10, batch_size=32, validation_split=0.2, callbacks=[TrainingProgressCallback()], class_weight=class_weights_dict)

model.save('emoji_prediction_model.h5')

evaluation_texts = ["I like pizza", "I hate pizza", "I went to china"]
evaluation_sequences = text_tokenizer.texts_to_sequences(evaluation_texts)
evaluation_padded = pad_sequences(evaluation_sequences, maxlen=text_padded.shape[1], padding='post')

predictions = model.predict(evaluation_padded)

for text, prediction in zip(evaluation_texts, predictions):
    top_n = 5  
    predicted_indices = np.argsort(prediction)[-top_n:][::-1]
    predicted_emojis = [emoji_tokenizer.index_word[idx] for idx in predicted_indices if idx != 0]
    print(f"Text: {text} -> Predicted Emojis: {predicted_emojis} with Probabilities: {[prediction[idx] for idx in predicted_indices if idx != 0]}")
