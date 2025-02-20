import streamlit as st
import numpy as np
import joblib
from pygam import LinearGAM

# --- Cargar el modelo GAM entrenado ---
gam = joblib.load('modelo_GAM.pkl')

# --- Interfaz de usuario en Streamlit ---
st.title("Predicción de Producción de Azúcar con GAM")

# Entradas del usuario
tcm = st.number_input("Ingrese Tcm:", min_value=0.0, format="%.2f")
rendimiento = st.number_input("Ingrese Rendimiento:", min_value=0.0, format="%.2f")
toneladas_jugo = st.number_input("Ingrese Toneladas de Jugo:", min_value=0.0, format="%.2f")

if st.button("Predecir Producción"):
    # --- Preparar los datos de entrada ---
    X_nuevo = np.array([[tcm, rendimiento, toneladas_jugo]])

    # --- Realizar la predicción ---
    y_pred_log = gam.predict(X_nuevo)

    # --- Invertir la transformación logarítmica (si el modelo fue entrenado en escala log) ---
    y_pred = np.expm1(y_pred_log)  # np.expm1() invierte np.log1p()

    # --- Mostrar el resultado ---
    st.success(f"⚡ Predicción de Producción: {y_pred[0]:,.2f} sacos")

