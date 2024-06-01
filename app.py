import streamlit as st
import requests
import emoji

st.title("Emoji Prediction App")
st.write("Enter text and get emoji predictions with probabilities")

text = st.text_input("Enter text:")

if st.button("Predict"):
    response = requests.post("http://localhost:8021/predict_emojis", json={"text": text})
    if response.status_code == 200:
        data = response.json()
        emojis = data['predicted_emojis']
        probabilities = data['probabilities']
        
        st.write("Predicted Emojis with Probabilities:")
        for emoji_instance, prob in zip(emojis, probabilities):
            emoji.emojize(f":{emoji_instance}:")
            st.write(f"{emoji_instance}: {prob:.2%}")
    else:
        st.write("Error: Unable to fetch predictions")


