import streamlit as st
import pandas as pd
import joblib
import numpy as np

def show_label_analysis(df):
    st.title("🎧 Aplikasi Prediksi Kesukaan Lagu Spotify")

    # Load model dan scaler
    model_path = "D:/SEMESTER 6/PENGGALIAN DATA/DASHBOARD SPOTIFY/model_spotify.pkl"
    scaler_path = "D:/SEMESTER 6/PENGGALIAN DATA/DASHBOARD SPOTIFY/scaler_spotify.pkl"

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # AUTO ambil fitur dari scaler (paling aman)
    fitur_model = list(scaler.feature_names_in_)

    # Fungsi auto align
    def align_features(df_input):
        df_input = df_input.copy()

        # tambah fitur yang kurang
        for col in fitur_model:
            if col not in df_input.columns:
                df_input[col] = 0.0

        # buang fitur yang tidak dipakai model
        df_input = df_input[fitur_model]

        return df_input.astype(float)

    mode = st.radio("Pilih Metode Input", ["Pilih Lagu", "Manual", "Upload .pkl"])

    # PILIH LAGU
    if mode == "Pilih Lagu":
        song_list = df[['track_name', 'artists']].drop_duplicates()
        song_list['display'] = song_list['track_name'] + " - " + song_list['artists']

        selected_song = st.selectbox("Pilih lagu:", song_list['display'].tolist())

        if selected_song:
            track = song_list[song_list['display'] == selected_song]['track_name'].values[0]
            artist = song_list[song_list['display'] == selected_song]['artists'].values[0]

            song_data = df[(df['track_name'] == track) & (df['artists'] == artist)]

            features = align_features(song_data)
            features_scaled = scaler.transform(features)

            pred = model.predict(features_scaled)[0]
            proba = model.predict_proba(features_scaled)[0]

            label = "Suka" if pred == 1 else "Tidak Suka"

            st.success(f"Prediksi: {label}")
            st.write(f"Probabilitas suka: {proba[1]*100:.2f}%")
            st.write(f"Probabilitas tidak suka: {proba[0]*100:.2f}%")

    # MANUAL INPUT (AUTO)
    elif mode == "Manual":
        st.write("Masukkan nilai fitur:")

        input_dict = {}

        for fitur in fitur_model:
            if fitur in ["danceability", "energy", "speechiness", "instrumentalness", "liveness", "valence"]:
                input_dict[fitur] = [st.slider(fitur, 0.0, 1.0, 0.5)]
            elif fitur == "tempo":
                input_dict[fitur] = [st.slider("tempo", 50.0, 200.0, 120.0)]
            elif fitur == "time_signature":
                input_dict[fitur] = [st.slider("time_signature", 3, 7, 4)]
            else:
                # fallback semua fitur lain
                input_dict[fitur] = [st.number_input(fitur, value=0.0)]

        input_df = pd.DataFrame(input_dict)

        features = align_features(input_df)
        features_scaled = scaler.transform(features)

        pred = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]

        label = "Suka" if pred == 1 else "Tidak Suka"

        st.success(f"Prediksi: {label}")
        st.write(f"Probabilitas suka: {proba[1]*100:.2f}%")
        st.write(f"Probabilitas tidak suka: {proba[0]*100:.2f}%")

    # UPLOAD PKL
    else:
        uploaded_file = st.file_uploader("Upload file .pkl", type=["pkl"])

        if uploaded_file is not None:
            try:
                uploaded_df = joblib.load(uploaded_file)

                if not isinstance(uploaded_df, pd.DataFrame):
                    st.error("File harus DataFrame")
                    return

                features = align_features(uploaded_df)
                features_scaled = scaler.transform(features)

                preds = model.predict(features_scaled)
                probas = model.predict_proba(features_scaled)

                results = []
                for i in range(len(preds)):
                    results.append({
                        "Prediksi": "Suka" if preds[i] == 1 else "Tidak Suka",
                        "Prob Suka (%)": f"{probas[i][1]*100:.2f}",
                        "Prob Tidak (%)": f"{probas[i][0]*100:.2f}"
                    })

                results_df = pd.DataFrame(results)

                st.write("Hasil:")
                st.dataframe(pd.concat([uploaded_df.reset_index(drop=True), results_df], axis=1))

            except Exception as e:
                st.error(f"Error: {e}")