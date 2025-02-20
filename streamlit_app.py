import streamlit as st
import numpy as np
import joblib
import os
import base64
from pygam import LinearGAM

# --- Cargar el modelo GAM entrenado ---
modelo_path = "modelo_GAM.pkl"
if os.path.exists(modelo_path):
    gam = joblib.load(modelo_path)
else:
    st.error("El archivo del modelo GAM no se encuentra. Verifica la ruta.")

# --- Cargar el logo si está disponible ---
logo_path = "logom.png"
if os.path.exists(logo_path):
    with open(logo_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    st.markdown(
        f'<div style="text-align: center;"><img src="data:image/png;base64,{encoded_image}" width="300"></div>',
        unsafe_allow_html=True
    )
else:
    st.warning("El logo no se encontró. Asegúrate de que el archivo esté en el directorio correcto.")

# --- Interfaz de usuario en Streamlit ---
st.title("MONTERREY AZUCARERA LOJANA")
st.subheader("Predicción de Producción de Azúcar con GAM")

st.write("""
Ingrese los valores en los campos a continuación para obtener una estimación de la producción de azúcar en sacos.
""")

# --- Entradas del usuario ---
tcm = st.number_input("Ingrese Tcm:", min_value=0.0, format="%.2f")
rendimiento = st.number_input("Ingrese Rendimiento:", min_value=0.0, format="%.2f")
toneladas_jugo = st.number_input("Ingrese Toneladas de Jugo:", min_value=0.0, format="%.2f")

# --- Botón para predecir ---
if st.button("Predecir Producción"):
    # --- Preparar los datos de entrada ---
    X_nuevo = np.array([[tcm, rendimiento, toneladas_jugo]])

    # --- Realizar la predicción ---
    y_pred_log = gam.predict(X_nuevo)

    # --- Invertir la transformación logarítmica ---
    y_pred = np.expm1(y_pred_log)  # np.expm1() invierte np.log1p()

    # --- Mostrar el resultado ---
    st.success(f"⚡ Predicción de Producción: {y_pred[0]:,.2f} sacos")
