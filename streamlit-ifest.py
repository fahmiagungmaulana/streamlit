import streamlit as st
import re
import pickle
import numpy as np
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import string

# Judul aplikasi
st.title('Aplikasi Prediksi Kalimat')

# Input teks dari pengguna
user_input = st.text_area('Masukkan kalimat yang akan diprediksi:', '')

# Fungsi untuk membersihkan input teks
def preprocess_text(text):
    text = re.sub(r'\[USERNAME]+', '', text)
    text = re.sub('[^\w\s]', '', text)
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("","",string.punctuation))
    text = text.strip()
    pisah = text.split()
    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()
    text =  stopword.remove(text)
    return text

# Tombol untuk memprediksi
if st.button('Prediksi'):
    cleaned_input = preprocess_text(user_input)
    if cleaned_input:
        # Memuat model yang telah difitkan sebelumnya
        model = tf.keras.models.load_model('modeltiga.h5')
        
        tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
        new_sequences = tokenizer.texts_to_sequences([cleaned_input])
        new_X = pad_sequences(new_sequences, maxlen=100)
        predictions = model.predict(new_X)


        label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))


        predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
        st.write('Hasil Prediksi:', predicted_labels)
    else:
        st.write('Teks input kosong. Masukkan kalimat untuk diprediksi.')
