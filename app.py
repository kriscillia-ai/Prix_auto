import streamlit as st
import numpy as np
import joblib

st.title("Prédiction du prix d'une voiture en fonction de ses caractéristiques")
st.subheader("Application réalisée par Kriscillia")
st.markdown("Cette application utilise un modèle de Machine Learning pour prédire le prix d'une voiture")

# Chargement du modèle
model = joblib.load(filename="final_model.joblib")

# Définition d'une fonction d'inférence
def inference(symboling, wheel_base, length, width, height, curb_weight, engine_size, compression_ratio, city_mpg, highway_mpg):
    new_data = np.array([
        symboling, wheel_base, length, width,
        height, curb_weight, engine_size, compression_ratio,
        city_mpg, highway_mpg
    ])
    pred = model.predict(new_data.reshape(1,-1))
    return pred

# L'utilisateur saisie une valeur pour chaque caractéristique de la voiture
symboling = st.number_input (label='symboling:', min_value=0, value=3)
wheel_base = st.number_input ("wheel_base:", value=90)
length = st.number_input ('length:', value=150)
width = st.number_input('width:', value=65)
height = st.number_input('height:', value=50)
curb_weight = st.number_input("curb-weight:", value=200)
engine_size = st.number_input ('engine-size:', value=120)
compression_ratio = st.number_input ('compression-ratio:', value=9.0)
city_mpg = st.number_input('city-mpg:', value=20)
highway_mpg = st.number_input ('highway-mpg:', value=30)

# Création du bouton "Predict" qui retourne la prédiction du modèle
if st.button('Predict'):
    prediction = inference(symboling, wheel_base, length, width, height, curb_weight, engine_size, compression_ratio, city_mpg, highway_mpg)
    resultat = "le prix en dollars de cette voiture est égal à : " + str(prediction[0])
    st.success(resultat)