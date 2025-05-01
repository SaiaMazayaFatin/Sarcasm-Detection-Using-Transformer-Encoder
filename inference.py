import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define your custom layers (TokenAndPositionEmbedding, TransformerEncoder) here
class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)
        self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, heads, neurons, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(neurons, activation="relu"), tf.keras.layers.Dense(embed_dim)]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dropout2 = tf.keras.layers.Dropout(0.5)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Load the trained model and tokenizer
custom_objects = {
    'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
    'TransformerEncoder': TransformerEncoder
}

loaded_model = tf.keras.models.load_model('model/sarcasm_detector_model.h5', custom_objects=custom_objects)

with open('model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Preprocessing input for the model
def preprocess_input(sentence, maxlen=20):
    sequences = tokenizer.texts_to_sequences([sentence])
    padded_sequences = pad_sequences(sequences, maxlen=maxlen)
    return padded_sequences

# Predict sarcasm
def predict_sarcasm(sentence):
    processed_input = preprocess_input(sentence)
    prediction = loaded_model.predict(processed_input)
    if prediction > 0.5:
        return "Sarcastic"
    else:
        return "Not Sarcastic"

# Streamlit UI
st.set_page_config(page_title="Sarcasm Detection", page_icon=":guardsman:", layout="wide")

# Title with a decorative header
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Sarcasm Detection App</h1>", unsafe_allow_html=True)

# Instructions
st.write("""
    ## Welcome to the Sarcasm Detection Application!
    This app uses a pre-trained model to predict whether a sentence is sarcastic or not. 
    Simply enter a sentence in the text box below and click "Predict" to check if the sentence is sarcastic.
""")

# Added clarification about language
st.markdown("""
    **Important:** This application currently only supports English text. Please enter sentences in English to get accurate predictions.
""", unsafe_allow_html=True)

# Input for the user with a modern design
user_input = st.text_area("Enter a Sentence", placeholder="Type your sentence here...", height=200, max_chars=500)

# Predict button with a modern look
if st.button("Predict", use_container_width=True):
    if user_input.strip():
        result = predict_sarcasm(user_input)
        # Display result with a stylish notification
        if result == "Sarcastic":
            st.success(f"Prediction: **{result}** 😏", icon="✅")
        else:
            st.info(f"Prediction: **{result}** 🙂", icon="ℹ️")
    else:
        st.warning("Please enter a sentence.", icon="⚠️")

# Footer with a nice touch
st.markdown("""
    <br><br>
    <footer style="text-align: center; color: gray;">
        Made with ❤️ by Saia Mazaya Fatin.<br>
        This app uses a deep learning model to predict sarcasm. <br>
        <a href="https://github.com/your-repository" target="_blank">GitHub Repository</a>
    </footer>
""", unsafe_allow_html=True)

