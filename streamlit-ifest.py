import streamlit as st
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Judul aplikasi
st.title('Aplikasi Prediksi Kalimat')

# Input teks dari pengguna
user_input = st.text_area('Masukkan kalimat yang akan diprediksi:', '')

# Fungsi untuk membersihkan input teks
def preprocess_text(text):
    text = re.sub(r'\[USERNAME]+', '', text)
    text = re.sub('[^\w\s]', '', text)
    return text

# Tombol untuk memprediksi
if st.button('Prediksi'):
    cleaned_input = preprocess_text(user_input)
    if cleaned_input:
        # Memuat model yang telah difitkan sebelumnya
        model = pickle.load(open('clf.pkl', 'rb'))

        # Memuat vektor TF-IDF yang diperlukan
        tfidf_vectorizer = TfidfVectorizer()

        # Membaca data vektor TF-IDF yang telah difitkan sebelumnya
        with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
            tfidf_vectorizer = pickle.load(vectorizer_file)

        # Menggunakan model dan vektor TF-IDF yang telah difitkan sebelumnya untuk membuat prediksi
        input_vector = tfidf_vectorizer.transform([cleaned_input])
        prediction = model.predict(input_vector)
        st.write('Hasil Prediksi:', prediction[0])
    else:
        st.write('Teks input kosong. Masukkan kalimat untuk diprediksi.')
