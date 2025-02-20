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
            background-color: #f7f9fb;
            font-family: 'Arial', sans-serif;
        }
        .title {
            text-align: center;
            font-size: 38px;
            color: #1a73e8;
            font-weight: 700;
        }
        .subtitle {
            text-align: center;
            font-size: 22px;
            color: #333333;
            margin-bottom: 20px;
            font-weight: 500;
        }
        .stButton>button {
            background-color: #8bc34a;  /* Verde suave */
            color: white;
            font-size: 18px;
            padding: 12px 20px;
            border-radius: 12px;
            transition: 0.3s;
            border: none;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        }
        .stButton>button:hover {
            background-color: #689f38;  /* Verde más oscuro */
            transform: scale(1.05);
        }
        .stNumberInput>div>div>input {
            font-size: 16px;
            text-align: center;
            padding: 10px;
            border-radius: 10px;
            border: 2px solid #1a73e8;
            box-shadow: 0px 2px 8px rgba(0, 0, 0, 0.1);
        }
        .result-box {
            background-color: #eaf3ff;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            font-size: 24px;
            font-weight: 600;
            color: #1a73e8;
            margin-top: 30px;
            border: 2px solid #1a73e8;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        }
        .stNumberInput>div>div>input:focus {
            border-color: #0050b3;
            outline: none;
        }
    </style>
""", unsafe_allow_html=True)

# --- Interfaz de usuario en Streamlit ---
st.markdown('<div class="title">MONTERREY AZUCARERA LOJANA</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predicción de Producción de Azúcar con GAM</div>', unsafe_allow_html=True)

st.write("""
Este aplicativo permite predecir la producción de azúcar a partir de tres variables clave:
- **Toneladas Caña Molida (TCM)**
- **Rendimiento (kg/TCM)**
- **Toneladas de Jugo**

La predicción se realiza mediante un **Modelo Aditivo Generalizado (GAM)**, un enfoque de machine learning que permite capturar relaciones no lineales entre las variables de entrada y la producción de azúcar. Este modelo ha sido entrenado con datos históricos del Ingenio Azucarero Monterrey C.A. y proporciona estimaciones basadas en patrones observados en la producción diaria.
""")

# --- Entrada de datos en formato CSV ---
st.write("Ingrese una lista de datos separados por comas en formato CSV, por ejemplo:")
st.write("`1000, 45.2, 800` (donde 1000 es TCM, 45.2 es rendimiento, y 800 son toneladas de jugo).")
input_data = st.text_area("Lista de Datos (Formato CSV)", "")

# --- Botón para generar predicciones ---
if st.button("Generar Predicciones"):
    if input_data.strip() == "":
        st.warning("⚠️ Por favor ingrese datos en formato CSV.")
    else:
        try:
            # --- Procesar los datos de entrada ---
            rows = input_data.strip().split("\n")
            data = [list(map(float, row.split(','))) for row in rows]

            # --- Realizar predicciones ---
            X_nuevo = np.array(data)
            y_pred_log = gam.predict(X_nuevo)

            # --- Invertir la transformación logarítmica ---
            y_pred = np.expm1(y_pred_log)  # np.expm1() invierte np.log1p()

            # --- Mostrar las predicciones ---
            st.markdown(f'<div class="result-box">⚡ Predicciones de Producción: </div>', unsafe_allow_html=True)
            for i, pred in enumerate(y_pred):
                st.markdown(f'**Entrada {i+1}:** {pred[0]:,.2f} sacos', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"⚠️ Error al procesar los datos. Verifica el formato de entrada. Detalles: {str(e)}")
