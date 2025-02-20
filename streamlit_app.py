import streamlit as st
import numpy as np
import joblib
from pygam import LinearGAM

# Configurar la página con icono y título
st.set_page_config(page_title="Predicción de Producción de Azúcar", page_icon="🍚")

# --- Cargar el modelo GAM entrenado ---
gam = joblib.load('modelo_GAM.pkl')

# --- Diseño de la interfaz ---
st.markdown(
    """
    <style>
        .main-container {
            background-color: #f4f4f4;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
            max-width: 600px;
            margin: auto;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            padding: 10px;
            border-radius: 10px;
        }
    </style>
    <div class='main-container'>
    """,
    unsafe_allow_html=True
)

# --- Título e información ---
st.title("📊 Predicción de Producción de Azúcar")

st.write("""
Este aplicativo permite predecir la producción de azúcar a partir de tres variables clave:
- **Toneladas Caña Molida (TCM)**
- **Rendimiento (kg/TCM)**
- **Toneladas de Jugo**

La predicción se realiza mediante un **Modelo Aditivo Generalizado (GAM)**, un enfoque de machine learning que permite capturar relaciones no lineales entre las variables de entrada y la producción de azúcar. Este modelo ha sido entrenado con datos históricos del Ingenio Azucarero Monterrey C.A. y proporciona estimaciones basadas en patrones observados en la producción diaria.
""")

st.write("""
Este aplicativo es útil para los profesionales e ingenieros azucareros de la empresa, ya que les permite obtener estimaciones de producción de manera rápida y basada en datos, facilitando la planificación y gestión operativa.
""")

# --- Entradas del usuario ---
tcm = st.number_input("Ingrese TCM (Toneladas Caña Molida):", min_value=0.0, format="%.2f")
rendimiento = st.number_input("Ingrese Rendimiento (kg/TCM):", min_value=0.0, format="%.2f")
toneladas_jugo = st.number_input("Ingrese Toneladas de Jugo:", min_value=0.0, format="%.2f")

# --- Validación y predicción ---
if st.button("Predecir Producción"):
    if tcm == 0.0 or rendimiento == 0.0 or toneladas_jugo == 0.0:
        st.warning("⚠️ Por favor, ingrese valores mayores a 0 en todos los campos.")
    else:
        X_nuevo = np.array([[tcm, rendimiento, toneladas_jugo]])
        y_pred_log = gam.predict(X_nuevo)
        y_pred = np.expm1(y_pred_log)  # Inversión de la transformación logarítmica
        st.success(f"⚡ Predicción de Producción: {y_pred[0]:,.2f} sacos")

# Cierre del diseño
st.markdown("</div>", unsafe_allow_html=True)
