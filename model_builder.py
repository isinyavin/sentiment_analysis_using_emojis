
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader, RandomSampler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


df = pd.read_csv('output_file.csv')
df['cleaned_text'] = df['cleaned_text'].astype(str).fillna('')
df['emojis'] = df['emojis'].astype(str).fillna('')

emoji_tokenizer = Tokenizer(num_words=500, filters='')
emoji_tokenizer.fit_on_texts(df['emojis'])
emoji_sequences = emoji_tokenizer.texts_to_sequences(df['emojis'])
emoji_padded = pad_sequences(emoji_sequences, padding='post')
emoji_labels = np.argmax(to_categorical(emoji_padded, num_classes=len(emoji_tokenizer.word_index) + 1), axis=1)

class_weights = compute_class_weight('balanced', classes=np.unique(emoji_labels), y=emoji_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

class EmojiDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

max_len = 128
texts = df['cleaned_text'].tolist()


train_dataset = EmojiDataset(
    texts=texts,
    labels=emoji_labels,
    tokenizer=bert_tokenizer,
    max_len=max_len
)

train_loader = DataLoader(
    train_dataset,
    sampler=RandomSampler(train_dataset),
    batch_size=32
)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(np.unique(emoji_labels)))
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=3e-5)

loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

epochs = 4
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits

        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_predictions += len(labels)

        if step % 5 == 0: 
            epoch_accuracy = correct_predictions.double() / total_predictions
            print(f"Step {step}, Loss: {loss.item()}, Epoch Accuracy So Far: {epoch_accuracy.item()}")

    avg_loss = total_loss / len(train_loader)
    epoch_accuracy = correct_predictions.double() / total_predictions
    print(f"Loss: {avg_loss}, Accuracy: {epoch_accuracy}")

model.save_pretrained('emoji_prediction_model_torch_revised')