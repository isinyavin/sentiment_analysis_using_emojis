import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import json
from fastapi import FastAPI, Request
from pydantic import BaseModel
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import json


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('emoji_prediction_model_torch_500')
model = model.to(device)
model.eval()


def load_emoji_tokenizer(file_path):
    with open(file_path, 'r') as file:
        tokenizer_config = json.load(file)
        tokenizer = Tokenizer(num_words=500, filters='')
        tokenizer.word_index = tokenizer_config['word_index']
        # Recreate index_word if it's missing
        if 'index_word' in tokenizer_config:
            tokenizer.index_word = tokenizer_config['index_word']
        else:
            tokenizer.index_word = {v: k for k, v in tokenizer.word_index.items()}
        return tokenizer

emoji_tokenizer = load_emoji_tokenizer('emoji_tokenizer_config_500torch.json')


index_to_emoji = {v: k for k, v in emoji_tokenizer.word_index.items()}


app = FastAPI()

class TextRequest(BaseModel):
    text: str

def preprocess_text(text, tokenizer, max_len):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    return encoding['input_ids'], encoding['attention_mask']

def predict_emojis(text, model, tokenizer, emoji_tokenizer):
    max_len = 128  
    input_ids, attention_mask = preprocess_text(text, tokenizer, max_len)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]

    top_n = 5
    top_indices = probabilities.argsort()[-top_n:][::-1]
    top_emojis = [index_to_emoji.get(i, '') for i in top_indices]
    top_probabilities = probabilities[top_indices]


    filtered_results = [(emoji, float(prob)) for emoji, prob in zip(top_emojis, top_probabilities) if emoji != '']

    return top_emojis, top_probabilities, filtered_results

@app.post("/predict_emojis")
async def predict_emojis_endpoint(request: TextRequest):
    text = request.text
    emojis, probabilities, filtered_results = predict_emojis(text, model, tokenizer, emoji_tokenizer)
    return {"text": text, "predicted_emojis": emojis, "probabilities": probabilities.tolist(), "filtered_results": filtered_results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8021)

