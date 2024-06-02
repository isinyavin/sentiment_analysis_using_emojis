# Applying Machine Learning/NLP Classification Techniques to Predict Relevant Emojis Given Textual Input

Trained a BERT (Bidirectional Encoder Representations from Transformers) model on 520K social media comments scraped from Reddit's API to develop a classification algorithm that predicts relevant emojis based on textual input, optimizing the model through adjustments of loss and optimization functions.

## Packages and Technologies Used

- **PyTorch**: For training and deploying the BERT model for inference.
- **Hugging Face Transformers**: For using the BERT model.
- **TensorFlow**: For preprocessing text data using a Tokenizer (and training older models found in older_model_files).
- **NumPy**: For numerical operations and handling arrays in working with model output.
- **pandas**: For data manipulation, cleaning, and analysis of scraped data before training.
- **Streamlit**: Used Streamlit to host the app and provide a clean UI.

## Files and Directories

- **individual_scrapes**: Contains CSV files of various downloads of comments.
- **emoji_prediction_model_torch_500**: The main machine learning model directory, which includes a classification model predicting 500 of the most popular emojis in training data.
- **data_cleaner.py**: Script for cleaning and optimizing input data for model training. This includes extracting and pairing emojis with text, and simplifying emojis by distilling optional emoji modifiers.
- **output_file.csv**: The main data file used for training the model.
- **older_model_files**: Contains various older models that perform worse than the current model. This includes an LSTM (Long Short-Term Memory) model and an unbalanced BERT model.
- **app.py**: The main Streamlit application script that runs the web interface for the emoji prediction model.
- **requirements.txt**: A file listing all the required Python packages for the project.

## How to Use the Model

Model is hosted at [https://predict-emojis.streamlit.app/]