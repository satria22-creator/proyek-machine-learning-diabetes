# app_diabetes.py

import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('model_diabetes_with_Pipeline.pkl')

# Judul aplikasi
st.title('Prediksi Risiko Diabetes')
st.write("Masukkan data berikut untuk memprediksi risiko diabetes Anda:")

# Input user
HighChol = st.selectbox('Apakah Anda memiliki kolesterol tinggi?', [0, 1])
CholCheck = st.selectbox('Apakah Anda pernah melakukan pemeriksaan kolesterol?', [0, 1])
BMI = st.number_input('BMI (Body Mass Index)', min_value=0.0, max_value=100.0, value=25.0)
HeartDiseaseorAttack = st.selectbox('Apakah Anda pernah mengalami serangan jantung atau penyakit jantung?', [0, 1])
PhysHlth = st.number_input('Jumlah hari dalam sebulan terakhir Anda merasa tidak sehat secara fisik', min_value=0, max_value=30, value=0)
DiffWalk = st.selectbox('Apakah Anda mengalami kesulitan berjalan atau naik tangga?', [0, 1])
HighBP = st.selectbox('Apakah Anda memiliki tekanan darah tinggi?', [0, 1])
Age_Lansia = st.selectbox('Apakah Anda berusia 60 tahun ke atas?', [0, 1])
GenHlth_Fair = st.selectbox('Apakah Anda merasa kesehatan Anda "cukup" baik?', [0, 1])
GenHlth_Poor = st.selectbox('Apakah Anda merasa kesehatan Anda "buruk"?', [0, 1])

# Prediksi ketika tombol diklik
if st.button('Prediksi'):
    # Masukkan data ke dalam array
    input_data = np.array([[HighChol, CholCheck, BMI, HeartDiseaseorAttack, PhysHlth,
                            DiffWalk, HighBP, Age_Lansia, GenHlth_Fair, GenHlth_Poor]])

    # Prediksi probabilitas
    probability = model.predict_proba(input_data)[0][1]  # Probabilitas kelas positif

    # Konversi ke persen
    probability_percent = probability * 100

    # Fungsi mapping kategori
    def map_probability_to_category(prob):
        if prob < 0.4:
            return 'Rendah'
        elif prob < 0.7:
            return 'Sedang'
        else:
            return 'Tinggi'

    # Tentukan kategori risiko
    risk_category = map_probability_to_category(probability)

    # Tampilkan hasil prediksi
    st.subheader('Hasil Prediksi:')
    st.write(f'**Probabilitas terkena diabetes:** {probability_percent:.2f}%')
    st.write(f'**Kategori Risiko:** {risk_category}')

    # Optional: Tambahkan saran umum
    if risk_category == 'Rendah':
        st.success('Risiko Anda rendah, pertahankan gaya hidup sehat! 💪')
    elif risk_category == 'Sedang':
        st.warning('Risiko Anda sedang, pertimbangkan untuk konsultasi dengan tenaga medis. ⚠️')
    else:
        st.error('Risiko Anda tinggi, segera konsultasikan dengan dokter. 🚨')
