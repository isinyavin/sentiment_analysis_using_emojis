from fastapi import FastAPI, Request
from pydantic import BaseModel
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import json

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("Using GPU")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    print("Using CPU")

model = TFBertForSequenceClassification.from_pretrained('emoji_prediction_model_torch')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def load_emoji_tokenizer(file_path):
    with open(file_path, 'r') as file:
        tokenizer_config = json.load(file)
        tokenizer = Tokenizer(num_words=300, filters='')
        tokenizer.word_index = tokenizer_config['word_index']
        
        if 'index_word' in tokenizer_config:
            tokenizer.index_word = tokenizer_config['index_word']
        else:
            tokenizer.index_word = {v: k for k, v in tokenizer.word_index.items()}
        return tokenizer

emoji_tokenizer = load_emoji_tokenizer('emoji_tokenizer_config2.json')

def predict_emojis(text, model, tokenizer, emoji_tokenizer):
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='tf', max_length=128)
    predictions = model(encoded_input).logits
    probabilities = tf.nn.softmax(predictions, axis=1).numpy()[0]

    
    top_indices = probabilities.argsort()[-5:][::-1]  
    top_emojis = [emoji_tokenizer.index_word.get(i, '') for i in top_indices]
    top_probabilities = probabilities[top_indices]

    
    filtered_results = [(emoji, float(prob)) for emoji, prob in zip(top_emojis, top_probabilities) if emoji != '']

    return filtered_results


app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.post("/predict_emojis")
async def predict_emojis_endpoint(request: TextRequest):
    text = request.text
    predictions = predict_emojis(text, model, bert_tokenizer, emoji_tokenizer)
    return {"text": text, "predictions": predictions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8015)
