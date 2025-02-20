import streamlit as st
import numpy as np
import joblib
from pygam import LinearGAM

# --- Cargar el modelo GAM entrenado ---
gam = joblib.load('modelo_GAM.pkl')  # Asegúrate de que el modelo esté en la misma carpeta del script

# --- Cargar el escalador (si se usó en el entrenamiento) ---
try:
    scaler = joblib.load("scaler.pkl")  # Cargar el escalador si se usó en el preprocesamiento
    use_scaler = True
except:
    use_scaler = False  # Si no se usó escalador, los datos se ingresan sin transformación

# --- Interfaz de usuario en Streamlit ---
st.title("Predicción de Producción de Azúcar con GAM")

# Entradas del usuario
tcm = st.number_input("Ingrese Tcm:", min_value=0.0, format="%.2f")
rendimiento = st.number_input("Ingrese Rendimiento:", min_value=0.0, format="%.2f")
toneladas_jugo = st.number_input("Ingrese Toneladas de Jugo:", min_value=0.0, format="%.2f")

if st.button("Predecir Producción"):
    # --- Preparar los datos de entrada ---
    X_nuevo = np.array([[tcm, rendimiento, toneladas_jugo]])

    # --- Aplicar escalado si se usó en el entrenamiento ---
    if use_scaler:
        X_nuevo = scaler.transform(X_nuevo)

    # --- Realizar la predicción ---
    y_pred_log = gam.predict(X_nuevo)

    # --- Deshacer la transformación logarítmica ---
    y_pred = np.expm1(y_pred_log)  # np.expm1() invierte la transformación log1p()

    # --- Mostrar el resultado ---
    st.success(f"⚡ Predicción de Producción: {y_pred[0]:,.2f} sacos")

