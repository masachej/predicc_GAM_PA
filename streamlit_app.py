import streamlit as st
import numpy as np
import joblib
import os
import base64
from pygam import LinearGAM

# --- Configurar la pestaña e icono usando el logo ---
logo_path = "logom.png"
if os.path.exists(logo_path):
    with open(logo_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    st.set_page_config(page_title="Predicción de Producción | MONTERREY", page_icon=f"data:image/png;base64,{encoded_image}", layout="centered")
else:
    st.set_page_config(page_title="Predicción de Producción | MONTERREY", page_icon="🌱", layout="centered")

# --- Cargar el modelo GAM entrenado ---
modelo_path = "modelo_GAM.pkl"
if os.path.exists(modelo_path):
    gam = joblib.load(modelo_path)
else:
    st.error("El archivo del modelo GAM no se encuentra. Verifica la ruta.")

# --- Cargar el logo si está disponible ---
if os.path.exists(logo_path):
    with open(logo_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    st.markdown(
        f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{encoded_image}" width="220" style="border-radius: 15px; margin-bottom: 20px;">
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
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #003366;
            transform: scale(1.05);
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
            margin-top: 20px;
            border: 2px solid #004d99;
            box-shadow: 3px 3px 10px rgba(0, 0, 0, 0.1);
        }
        .content-container {
            border: 4px solid #4CAF50;  /* Cambiar color del borde a verde */
            padding: 20px;
            border-radius: 15px;
            background-color: #ffffff;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# --- Interfaz de usuario en Streamlit ---
st.markdown('<div class="title">MONTERREY AZUCARERA LOJANA</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predicción de Producción de Azúcar con GAM</div>', unsafe_allow_html=True)

# --- Contenedor con marco verde alrededor del contenido principal ---
with st.container():
    st.markdown('<div class="content-container">', unsafe_allow_html=True)

    st.write("""
    Este aplicativo permite predecir la producción de azúcar a partir de tres variables clave:
    - **Toneladas Caña Molida (TCM)**
    - **Rendimiento (kg/TCM)**
    - **Toneladas de Jugo**

    La predicción se realiza mediante un **Modelo Aditivo Generalizado (GAM)**, un enfoque de machine learning que permite capturar relaciones no lineales entre las variables de entrada y la producción de azúcar. Este modelo ha sido entrenado con datos históricos del Ingenio Azucarero Monterrey C.A. y proporciona estimaciones basadas en patrones observados en la producción diaria.
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

            # --- Mostrar el resultado con un marco elegante ---
            st.markdown(f'<div class="result-box">⚡ Predicción de Producción: {y_pred[0]:,.2f} sacos</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
