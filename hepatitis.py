import pickle
import streamlit as st
import pandas as pd

model = pickle.load(open("model_hepatitis.sav", 'rb'))

st.title("Memprediksi Penyakit Hepatitis")
age = st.number_input("Masukkan umur")

sex_options = ['Laki - Laki', 'Perempuan']
sex = st.selectbox('Masukkan jenis kelamin', sex_options)
if sex == 'Laki - Laki':
    sex = 1
else:
    sex = 2

steroid = st.selectbox("Apakah obat steroid benar atau tidak?", [True, False])
antivirals = st.selectbox(
    "Apakah obat antiviral benar atau tidak?", [True, False])
fatigue = st.selectbox(
    "Apakah gejala fatigue benar atau tidak?", [True, False])
malaise = st.selectbox(
    "Apakah gejala malaise benar atau tidak?", [True, False])
anorexia = st.selectbox(
    "Apakah gejala anorexia benar atau tidak?", [True, False])
liver_big = st.selectbox(
    "Apakah hati membesar benar atau tidak?", [True, False])
liver_firm = st.selectbox(
    "Apakah hati terasa keras benar atau tidak?", [True, False])
spleen_palpable = st.selectbox(
    "Apakah limpa teraba benar atau tidak?", [True, False])
spiders = st.selectbox(
    "Apakah spider angioma benar atau tidak?", [True, False])
ascites = st.selectbox("Apakah ascites benar atau tidak?", [True, False])
varices = st.selectbox("Apakah varices benar atau tidak?", [True, False])
bilirubin = st.number_input("Masukkan kadar bilirubin")
alk_phosphate = st.number_input("Masukkan kadar alk_phosphate")
sgot = st.number_input("Masukkan kadar sgot")
albumin = st.number_input("Masukkan kadar albumin")
histology = st.selectbox("Apakah histology benar atau tidak?", [True, False])

predict = ""

if st.button("hepatitis"):
    input_data = [[age, sex, steroid, antivirals, fatigue, malaise, anorexia,
                  liver_big, liver_firm, spleen_palpable, spiders, ascites, varices, bilirubin,
                  alk_phosphate, sgot, albumin, histology]]
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        result = "Hidup"
    else:
        result = "Mati"
    st.write(f"hepatitis dalam predict : {result}")
