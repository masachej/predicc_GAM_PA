import numpy as np
# Aplicar monkey patch para evitar el error de np.int
if not hasattr(np, 'int'):
    np.int = int

import streamlit as st
import joblib
import os
import base64
from pygam import LinearGAM

# Configurar la página
st.set_page_config(
    page_title="Predicción de Producción - Monterrey Azucarera Lojana",
    page_icon="🌾",
    layout="centered"
)

# --- Cargar el modelo GAM ---
modelo_path = "modelo_GAM.pkl"
if os.path.exists(modelo_path):
    modelo_gam = joblib.load(modelo_path)
else:
    st.error("🚨 El archivo del modelo GAM no se encuentra. Verifica la ruta.")

# --- Cargar el escalador ---
scaler_path = "scaler.pkl"
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
else:
    st.error("🚨 El archivo del escalador no se encuentra. Verifica la ruta.")

# --- Función para hacer la predicción ---
def make_prediction(tcm, rendimiento, toneladas_jugo):
    try:
        # Preparar los datos de entrada
        data = np.array([[tcm, rendimiento, toneladas_jugo]])
        data_scaled = scaler.transform(data)  # Escalar los datos
        prediction_log = modelo_gam.predict(data_scaled)  # Hacer la predicción en escala logarítmica
        prediction = np.expm1(prediction_log)  # Deshacer la transformación logarítmica
        return prediction[0]  # Devolver la predicción como un solo valor
    except Exception as e:
        st.error(f"⚠️ Ocurrió un error en la predicción: {e}")
        return None

# --- Cargar y mostrar el logo ---
logo_path = "logom.png"
if os.path.exists(logo_path):
    with open(logo_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    st.markdown(
        f'<div style="text-align: center;"><img src="data:image/png;base64,{encoded_image}" width="250"></div>',
        unsafe_allow_html=True
    )
else:
    st.warning("⚠️ El logo no se encontró. Asegúrate de que el archivo esté en el directorio correcto.")

# --- Títulos y descripción ---
st.markdown(
    """
    <h1 style='text-align: center; color: #2E8B57;'>Monterrey Azucarera Lojana</h1>
    <h3 style='text-align: center; color: #555;'>Predicción de Producción de Azúcar</h3>
    <p style='text-align: center; font-size:16px;'>
        Este aplicativo permite predecir la producción de azúcar a partir de tres variables clave:
        <ul>
            <li><b>Toneladas Caña Molida (TCM)</b></li>
            <li><b>Rendimiento (kg/TCM)</b></li>
            <li><b>Toneladas de Jugo</b></li>
        </ul>
    </p>
    <hr style="border: 1px solid #ddd;">
    """,
    unsafe_allow_html=True
)

# --- Entradas de usuario ---
st.subheader("Ingrese los valores para realizar la predicción:")
col1, col2, col3 = st.columns(3)

with col1:
    tcm = st.number_input("📌 Toneladas Caña Molida (ton)", min_value=0.0, value=0.0, step=0.01)
with col2:
    rendimiento = st.number_input("📌 Rendimiento (kg/TCM)", min_value=0.0, value=0.0, step=0.01)
with col3:
    toneladas_jugo = st.number_input("📌 Toneladas de Jugo (ton)", min_value=0.0, value=0.0, step=0.01)

# --- Botón para predecir ---
if st.button("🔍 Realizar Predicción"):
    if tcm == 0.0 or rendimiento == 0.0 or toneladas_jugo == 0.0:
        st.warning("⚠️ Por favor, ingrese valores mayores a 0 en todos los campos.")
    else:
        result = make_prediction(tcm, rendimiento, toneladas_jugo)
        if result is not None:
            st.success(f"✅ La predicción de producción es: **{result:,.2f} sacos**.")

# --- Estilos adicionales ---
st.markdown(
    """
    <style>
        .stButton>button {
            background-color: #2E8B57;
            color: white;
            font-size: 16px;
            border-radius: 10px;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #228B22;
        }
    </style>
    """,
    unsafe_allow_html=True
)
