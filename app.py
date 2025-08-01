import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import streamlit as st
with st.spinner('Loading model and tokenizer...'):
    import torch
    from transformers import BertTokenizer, BertForSequenceClassification
    import json
    from keras.preprocessing.text import Tokenizer
    import numpy as np
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource 
def load_model_and_tokenizers():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('emoji_prediction_model_torch_500')
    model = model.to(device)
    model.eval()

    def load_emoji_tokenizer(file_path):
        with open(file_path, 'r') as file:
            tokenizer_config = json.load(file)
            tokenizer = Tokenizer(num_words=500, filters='')
            tokenizer.word_index = tokenizer_config['word_index']
            if 'index_word' in tokenizer_config:
                tokenizer.index_word = tokenizer_config['index_word']
            else:
                tokenizer.index_word = {v: k for k, v in tokenizer.word_index.items()}
            return tokenizer

    emoji_tokenizer = load_emoji_tokenizer('emoji_tokenizer_config_500torch.json')
    return tokenizer, model, emoji_tokenizer

@st.cache_data 
def get_index_to_emoji(_emoji_tokenizer):
    return {v: k for k, v in _emoji_tokenizer.word_index.items()}

tokenizer, model, emoji_tokenizer = load_model_and_tokenizers()
index_to_emoji = get_index_to_emoji(emoji_tokenizer)

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

@st.cache_data  
def predict_emojis(text, _model, _tokenizer, _emoji_tokenizer, _index_to_emoji):
    max_len = 128  
    input_ids, attention_mask = preprocess_text(text, _tokenizer, max_len)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = _model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]

    top_n = 5
    top_indices = probabilities.argsort()[-top_n:][::-1]
    top_emojis = [_index_to_emoji.get(i, '') for i in top_indices]
    top_probabilities = probabilities[top_indices]

    filtered_results = [(emoji, float(prob)) for emoji, prob in zip(top_emojis, top_probabilities) if emoji != '']

    return top_emojis, top_probabilities, filtered_results


st.title("Emoji Prediction Model")

text = st.text_input("Enter text to predict emojis:")
if st.button("Predict"):
    if text:
        with st.spinner('Predicting...'):
            emojis, probabilities, filtered_results = predict_emojis(text, model, tokenizer, emoji_tokenizer, index_to_emoji)
            for emoji, prob in filtered_results:
                st.write(f"{emoji}: {prob:.2%}")
    else:
        st.write("Please enter some text to predict emojis.")



