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
        f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{encoded_image}" width="250" style="border-radius: 15px;">
        </div>
        """,
        unsafe_allow_html=True
    )

# --- Aplicar estilos CSS personalizados ---
st.markdown("""
    <style>
        body {
            background-color: #f4f4f4;
            font-family: 'Arial', sans-serif;
        }
        .title {
            text-align: center;
            font-size: 36px;
            color: #004d99;
            font-weight: bold;
        }
        .subtitle {
            text-align: center;
            font-size: 20px;
            color: #666;
            margin-bottom: 20px;
        }
        .stButton>button {
            background-color: #004d99;
            color: white;
            font-size: 18px;
            padding: 10px;
            border-radius: 10px;
        }
        .stButton>button:hover {
            background-color: #003366;
        }
        .stNumberInput>div>div>input {
            text-align: center;
        }
        .result-box {
            background-color: #e6f2ff;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-size: 22px;
            font-weight: bold;
            color: #004d99;
        }
    </style>
""", unsafe_allow_html=True)

# --- Interfaz de usuario en Streamlit ---
st.markdown('<div class="title">MONTERREY AZUCARERA LOJANA</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predicción de Producción de Azúcar con GAM</div>', unsafe_allow_html=True)

st.write("""
Ingrese los valores en los campos a continuación para obtener una estimación de la producción de azúcar en sacos.
""")

# --- Sección de entrada de datos ---
col1, col2, col3 = st.columns(3)
with col1:
    tcm = st.number_input("Ingrese Tcm:", min_value=0.0, format="%.2f")
with col2:
    rendimiento = st.number_input("Ingrese Rendimiento:", min_value=0.0, format="%.2f")
with col3:
    toneladas_jugo = st.number_input("Ingrese Toneladas de Jugo:", min_value=0.0, format="%.2f")

# --- Botón para predecir ---
if st.button("Predecir Producción"):
    if tcm <= 0.0 or rendimiento <= 0.0 or toneladas_jugo <= 0.0:
        st.warning("⚠️ Por favor, ingrese valores mayores a 0 en todos los campos antes de predecir.")
    else:
        # --- Preparar los datos de entrada ---
        X_nuevo = np.array([[tcm, rendimiento, toneladas_jugo]])

        # --- Realizar la predicción ---
        y_pred_log = gam.predict(X_nuevo)

        # --- Invertir la transformación logarítmica ---
        y_pred = np.expm1(y_pred_log)  # np.expm1() invierte np.log1p()

        # --- Mostrar el resultado con diseño mejorado ---
        st.markdown(f'<div class="result-box">⚡ Predicción de Producción: {y_pred[0]:,.2f} sacos</div>', unsafe_allow_html=True)
