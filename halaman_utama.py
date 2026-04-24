import streamlit as st
import pandas as pd
import joblib

from feature_importance import show_feature_importance_page
from overview import show_overview
from song_recommendation import show_song_recommendation
from favorite_song_prediction import show_label_analysis


@st.cache_data
def load_data():
    return pd.read_csv("dataset_final.csv")


df = load_data()
model = joblib.load("model_spotify.pkl")


st.sidebar.title("Spotify Dashboard")

page = st.sidebar.radio(
    "Pilih Halaman",
    [
        "Overview Dashboard",
        "Aplikasi Prediksi Kesukaan Lagu Spotify",
        "Rekomendasi Lagu",
        "Feature Importance"
    ]
)


if page == "Overview Dashboard":
    show_overview(df)

elif page == "Aplikasi Prediksi Kesukaan Lagu Spotify":
    show_label_analysis(df)

elif page == "Feature Importance":
    show_feature_importance_page(df, model)

elif page == "Rekomendasi Lagu":
    show_song_recommendation(df)

else:
    st.title(f"Halaman {page} sedang dalam pengembangan")