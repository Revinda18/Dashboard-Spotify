import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


def show_feature_importance_page(df, model):
    st.title("Feature Importance")

    top_n = st.slider(
        "Pilih jumlah fitur teratas:",
        min_value=5,
        max_value=30,
        value=10
    )

    # 🔥 Ambil jumlah fitur dari model
    if hasattr(model, 'feature_importances_'):
        n_features = len(model.feature_importances_)
    elif hasattr(model, 'coef_'):
        coef = model.coef_
        n_features = coef.shape[1] if coef.ndim > 1 else len(coef)
    else:
        st.error("Model tidak punya feature importance")
        st.stop()

    # 🔥 Ambil fitur dari dataframe (hanya kolom numerik biar aman)
    feature_names = df.select_dtypes(include=np.number).columns.tolist()

    # 🔥 Sinkronisasi jumlah fitur
    if len(feature_names) != n_features:
        st.warning(f"""
        ⚠️ Jumlah fitur tidak sama:
        - DataFrame: {len(feature_names)}
        - Model: {n_features}

        👉 Akan disesuaikan otomatis (ambil {n_features} fitur pertama)
        """)

        feature_names = feature_names[:n_features]

    # 🔥 Plot
    feat_imp = plot_feature_importance(model, feature_names, top_n)
    st.dataframe(feat_imp)


def plot_feature_importance(model, feature_names, top_n=20):

    # Ambil importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_

    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
        if importances.ndim == 2:
            importances = importances.mean(axis=0)

    else:
        raise AttributeError("Model tidak memiliki feature_importances_ atau coef_")

    # Final check (harus sama sekarang)
    if len(feature_names) != len(importances):
        st.error("❌ Gagal sinkronisasi fitur dengan model")
        st.stop()

    # DataFrame
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })

    feat_imp = feat_imp.sort_values(by='importance', ascending=False).head(top_n)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feat_imp['feature'][::-1], feat_imp['importance'][::-1])
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Top {top_n} Feature Importance')

    st.pyplot(fig)
    plt.close(fig)

    return feat_imp